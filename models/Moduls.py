import os
import torch
import pandas as pd
import yaml
from models.Network.CNNLSTMWithAttention import *


class Modul:
    def __init__(self, model_params):
        self.model_params = model_params
        model_class = eval(model_params['model'])
        self.model = model_class(model_params)

        self.loss_list = []
        self.ic_list = []
        self.pred_start_ic = 0.12
        self.cur_ic_train = -1

        output_path = os.path.join(model_params['output_path'], model_params['config_name'])
        os.makedirs(output_path, exist_ok=True)
        # 模型保存路径
        model_path = os.path.join(output_path, 'Model')
        os.makedirs(model_path, exist_ok=True)
        # 预测值保存路径
        prediction_path = os.path.join(output_path, 'Prediction')
        os.makedirs(prediction_path, exist_ok=True)
        # 记录loss和ic值
        metrics_df_path = os.path.join(output_path, 'metrics_df.xlsx')
        predict_metrics_path = os.path.join(output_path, 'predict_metrics')

        self.model_path = model_path
        self.prediction_path = prediction_path
        self.metrics_df_path = metrics_df_path
        self.predict_metrics_path = predict_metrics_path

        self.metrics_df = pd.DataFrame(columns=['loss_train', 'ic_train', 'loss_pred', 'ic_pred'], index=range(1, 500))
        if os.path.exists(self.metrics_df_path):
            self.metrics_df.update(pd.read_excel(self.metrics_df_path, index_col=0))
        self.predict_metrics = {i: {'ic_list': {}, 'loss_list': {}} for i in range(1, 500)}

        self.lr = model_params['lr']
        self.lr_decay = model_params['lr_decay']
        self.l1_norm = model_params['l1_norm']
        self.loss = model_params['loss']

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lr_decay)

        if 'is_schedule' in model_params and model_params['is_schedule']:
            self.is_schedule = True
        else:
            self.is_schedule = False

        if self.is_schedule:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        if self.loss == 'mse':
            self.mse_loss = torch.nn.MSELoss(reduce='mean')
            self.loss_fun = self.cal_mse_loss
        elif 'huber_loss' in self.loss:
            delta = int(self.loss.split('_')[-1])
            self.huber_loss = torch.nn.HuberLoss(reduction='mean', delta=delta)
            self.loss_fun = self.cal_huber_loss
        elif 'weighted_mse' in self.loss:
            self.weighted_threshold = float(self.loss.split('_')[-1])
            self.mse_loss_no_reduce = torch.nn.MSELoss(reduction='none')
            self.loss_fun = self.cal_weighted_mse_loss
        else:
            raise Exception('loss function not implement')

    def cal_mse_loss(self, y_true, y_pred):
        return self.mse_loss(y_true, y_pred)

    def cal_weighted_mse_loss(self, y_true, y_pred):
        weights = torch.where(torch.abs(y_true) > self.weighted_threshold, 1.0, 0.1)
        weights = weights.type(torch.float32)
        loss = self.mse_loss_no_reduce(y_true, y_pred)
        weighted_loss = (loss * weights).sum() / weights.sum()
        return weighted_loss

    def cal_huber_loss(self, y_true, y_pred):
        return self.huber_loss(y_true, y_pred)

    def cal_ic(self, x, y):
        return (((x - x.mean()) / x.std()) * ((y - y.mean()) / y.std())).mean()

    def Train_batch(self, data_batch, label_batch):
        self.model.train()
        loss_fun = self.loss_fun

        x, px = data_batch
        x = torch.from_numpy(x).cpu()
        px = torch.from_numpy(px).cpu()
        label_batch = torch.from_numpy(label_batch).cpu()
        x = x.float()
        px = px.float()
        label_batch = label_batch.float()
        data_batch = (x, px)
        self.optimizer.zero_grad()
        output_batch = self.model(data_batch)

        loss = loss_fun(label_batch, output_batch)

        if self.l1_norm > 0:
            loss += self.l1_norm * sum([p.abs().sum() for p in self.model.parameters()])

        if torch.isnan(loss):
            raise Exception('Nan loss exists')
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
        self.optimizer.step()

        with torch.no_grad():
            loss = loss_fun(label_batch[:], output_batch[:])
            ic = self.cal_ic(label_batch[:].cpu().numpy(), output_batch[:].cpu().numpy())
            self.loss_list.append(loss.item())
            self.ic_list.append(ic.item())

    def Train_epoch_finish(self, epoch):
        cur_epoch = str(epoch).zfill(2)
        if self.is_schedule:
            self.scheduler.step()
        cur_epoch_loss = np.nanmean(self.loss_list)
        cur_epoch_ic = np.nanmean(self.ic_list)
        self.cur_ic_train = cur_epoch_ic

        self.loss_list = []
        self.ic_list = []

        torch.save(self.model.state_dict(), os.path.join(self.model_path, f'model_{cur_epoch}'))
        self.metrics_df.loc[epoch, ['loss_train', 'ic_train']] = [cur_epoch_loss, cur_epoch_ic]
        self.metrics_df.to_excel(self.metrics_df_path)

    def Predict_date(self, data, label, keys, date):
        ic_train = self.metrics_df['ic_train']
        ic_train = ic_train.dropna()
        ic_train = ic_train[ic_train > self.pred_start_ic]
        x, px = data
        x = torch.from_numpy(x).cpu()
        px = torch.from_numpy(px).cpu()
        label = torch.from_numpy(label).cpu()
        x = x.float()
        px = px.float()
        label = label.float()
        data = (x, px)
        for epoch in range(ic_train.index[0], ic_train.index[-1] + 1):
            cur_epoch = str(epoch).zfill(2)
            model = self.model
            loss_fun = self.loss_fun
            os.makedirs(os.path.join(self.prediction_path, cur_epoch), exist_ok=True)

            with torch.no_grad():
                model.load_state_dict(
                    torch.load(os.path.join(self.model_path, f'model_{cur_epoch}'), map_location=torch.device('cpu'), weights_only=True))
                model.eval()

                output_pred = model(data)
                # output_pred = output_pred.cpu().numpy()

            # output_pred = torch.from_numpy(output_pred.values)
            with torch.no_grad():
                loss = loss_fun(label, output_pred)

            ic = self.cal_ic(label, output_pred)

            self.predict_metrics[epoch]['loss_list'][date] = loss.item()
            self.predict_metrics[epoch]['ic_list'][date] = ic.item()

            pd.to_pickle(self.predict_metrics, self.predict_metrics_path)
            pd.to_pickle(pd.DataFrame(output_pred, index=keys), os.path.join(self.prediction_path,cur_epoch,f'{date}.pkl'))

    def Predict_finish(self):
        ic_train = self.metrics_df['ic_train']
        ic_train = ic_train.dropna()
        ic_train = ic_train[ic_train > self.pred_start_ic]
        for epoch in range(ic_train.index[0], ic_train.index[-1] + 1):
            loss_list = list(self.predict_metrics[epoch]['loss_list'].values())
            ic_list = list(self.predict_metrics[epoch]['ic_list'].values())

            cur_epoch_loss = np.nanmean(loss_list)
            cur_epoch_ic = np.nanmean(ic_list)
            self.metrics_df.loc[epoch, ['loss_pred', 'ic_pred']] = [cur_epoch_loss, cur_epoch_ic]
            epoch += 1
            self.metrics_df.to_excel(self.metrics_df_path)