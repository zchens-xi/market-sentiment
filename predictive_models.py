# -*- coding: utf-8 -*-
"""
预测模型模块
包含SARIMAX、Transformer、LSTM和LSTNet等多种时间序列预测模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# 深度学习框架
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch未安装，将跳过深度学习模型")

# 时间序列模型
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ Statsmodels未安装，将跳过SARIMAX模型")

import config
import os
from datetime import datetime, timedelta


class ModelEvaluator:
    """模型评估器"""

    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, y_true, y_pred, model_name):
        """计算评估指标"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # 避免除零错误
        y_true_nonzero = y_true[y_true != 0]
        y_pred_nonzero = y_pred[y_true != 0]

        if len(y_true_nonzero) > 0:
            mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
        else:
            mape = np.inf

        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = -np.inf

        self.metrics[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2
        }

        return self.metrics[model_name]

    def print_metrics(self, model_name):
        """打印评估指标"""
        if model_name not in self.metrics:
            print(f"❌ 模型 {model_name} 的指标尚未计算")
            return

        metrics = self.metrics[model_name]
        print(f"\n📊 {model_name} 模型评估指标:")
        print("="*50)
        print(f"MSE (均方误差):     {metrics['MSE']:.6f}")
        print(f"RMSE (均方根误差):  {metrics['RMSE']:.6f}")
        print(f"MAE (平均绝对误差): {metrics['MAE']:.6f}")
        print(f"MAPE (平均绝对百分比误差): {metrics['MAPE']:.2f}%")
        print(f"R² (决定系数):      {metrics['R²']:.4f}")
        print("="*50)

    def compare_models(self):
        """比较所有模型"""
        if not self.metrics:
            print("❌ 没有模型指标可供比较")
            return

        print("\n🏆 模型性能比较")
        print("="*80)

        # 创建比较表格
        comparison_df = pd.DataFrame(self.metrics).T
        comparison_df = comparison_df.round(6)

        # 排序（按RMSE升序）
        comparison_df = comparison_df.sort_values('RMSE')

        print(comparison_df.to_string())

        # 找出最佳模型
        best_model = comparison_df.index[0]
        print(f"\n🥇 最佳模型: {best_model}")
        print(f"   RMSE: {comparison_df.loc[best_model, 'RMSE']:.6f}")
        print(f"   R²: {comparison_df.loc[best_model, 'R²']:.4f}")

        return comparison_df


class SARIMAXModel:
    """SARIMAX时间序列模型"""

    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.scaler = StandardScaler()

    def prepare_data(self, df, target_col='日收益率', exog_cols=None):
        """准备SARIMAX数据"""
        # 确保数据按时间排序
        df_sorted = df.sort_index()

        # 目标变量
        y = df_sorted[target_col].dropna()

        # 外生变量
        if exog_cols:
            exog = df_sorted[exog_cols].loc[y.index].fillna(method='ffill').fillna(0)
            # 标准化外生变量
            exog_scaled = pd.DataFrame(
                self.scaler.fit_transform(exog),
                index=exog.index,
                columns=exog.columns
            )
            return y, exog_scaled
        else:
            return y, None

    def auto_arima_params(self, y, seasonal=True):
        """自动确定ARIMA参数"""
        if not STATSMODELS_AVAILABLE:
            return (1, 1, 1), (1, 1, 1, 12) if seasonal else None

        from statsmodels.tsa.arima.model import ARIMA
        from itertools import product

        # 简化的网格搜索
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)

        best_aic = float('inf')
        best_params = (1, 1, 1)

        for p, d, q in product(p_values, d_values, q_values):
            try:
                model = ARIMA(y, order=(p, d, q))
                fitted_model = model.fit()
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_params = (p, d, q)
            except:
                continue

        seasonal_params = (1, 1, 1, 12) if seasonal else None
        return best_params, seasonal_params

    def fit(self, y, exog=None, order=None, seasonal_order=None, auto_params=True):
        """训练SARIMAX模型"""
        if not STATSMODELS_AVAILABLE:
            print("❌ Statsmodels未安装，无法使用SARIMAX模型")
            return None

        print("🚀 开始训练SARIMAX模型...")

        # 自动确定参数
        if auto_params or order is None:
            print("📊 自动确定ARIMA参数...")
            order, seasonal_order = self.auto_arima_params(y, seasonal=seasonal_order is not None)
            print(f"选择的参数: order={order}, seasonal_order={seasonal_order}")

        try:
            # 创建并拟合模型
            self.model = SARIMAX(
                y,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            self.fitted_model = self.model.fit(disp=False)
            print("✅ SARIMAX模型训练完成")

            # 打印模型摘要
            print("\n📋 模型摘要:")
            print(self.fitted_model.summary().tables[1])

            return self.fitted_model

        except Exception as e:
            print(f"❌ SARIMAX模型训练失败: {e}")
            return None

    def predict(self, steps=30, exog_future=None):
        """预测未来值"""
        if self.fitted_model is None:
            print("❌ 模型尚未训练")
            return None

        try:
            forecast = self.fitted_model.forecast(steps=steps, exog=exog_future)
            conf_int = self.fitted_model.get_forecast(steps=steps, exog=exog_future).conf_int()

            return {
                'forecast': forecast,
                'conf_int': conf_int,
                'lower_bound': conf_int.iloc[:, 0],
                'upper_bound': conf_int.iloc[:, 1]
            }
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None


# PyTorch模型类
if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """LSTM模型"""

        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )

            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # LSTM前向传播
            lstm_out, _ = self.lstm(x)

            # 取最后一个时间步的输出
            out = self.dropout(lstm_out[:, -1, :])
            out = self.fc(out)

            return out


    class TransformerModel(nn.Module):
        """Transformer模型"""

        def __init__(self, input_size, d_model=64, nhead=8, num_layers=3, dropout=0.1, output_size=1):
            super(TransformerModel, self).__init__()
            self.d_model = d_model

            # 输入投影层
            self.input_projection = nn.Linear(input_size, d_model)

            # 位置编码
            self.pos_encoding = PositionalEncoding(d_model, dropout)

            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # 输出层
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(d_model, output_size)

        def forward(self, x):
            # 输入投影
            x = self.input_projection(x)

            # 添加位置编码
            x = self.pos_encoding(x)

            # Transformer编码
            transformer_out = self.transformer(x)

            # 取最后一个时间步
            out = self.dropout(transformer_out[:, -1, :])
            out = self.fc(out)

            return out


    class PositionalEncoding(nn.Module):
        """位置编码"""

        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0), :].transpose(0, 1)
            return self.dropout(x)


    class LSTNetModel(nn.Module):
        """LSTNet模型 (Long- and Short-term Time-series network)"""

        def __init__(self, input_size, conv_channels=100, kernel_size=6,
                     lstm_hidden=100, dropout=0.2, skip=24, highway_window=24, output_size=1):
            super(LSTNetModel, self).__init__()

            self.skip = skip
            self.highway_window = highway_window

            # 卷积层
            self.conv1d = nn.Conv1d(input_size, conv_channels, kernel_size=kernel_size)
            self.dropout = nn.Dropout(dropout)

            # LSTM层
            self.lstm = nn.LSTM(conv_channels, lstm_hidden, batch_first=True)

            # Skip连接
            if skip > 0:
                self.skip_lstm = nn.LSTM(conv_channels, lstm_hidden, batch_first=True)
                self.skip_linear = nn.Linear(lstm_hidden, output_size)

            # Highway网络
            self.highway = nn.Linear(highway_window, output_size)

            # 输出层
            self.output = nn.Linear(lstm_hidden, output_size)

        def forward(self, x):
            # x shape: (batch_size, seq_len, input_size)
            batch_size, seq_len, input_size = x.shape

            # 卷积层 (需要转置为 (batch_size, input_size, seq_len))
            conv_input = x.transpose(1, 2)
            conv_out = torch.relu(self.conv1d(conv_input))
            conv_out = self.dropout(conv_out)

            # 转回为 (batch_size, seq_len, conv_channels)
            conv_out = conv_out.transpose(1, 2)

            # LSTM层
            lstm_out, _ = self.lstm(conv_out)
            lstm_out = self.dropout(lstm_out[:, -1, :])  # 取最后一个输出

            # 主要输出
            main_out = self.output(lstm_out)

            # Skip连接
            skip_out = 0
            if self.skip > 0 and conv_out.size(1) >= self.skip:
                skip_conv = conv_out[:, -self.skip:, :]
                skip_lstm_out, _ = self.skip_lstm(skip_conv)
                skip_out = self.skip_linear(skip_lstm_out[:, -1, :])

            # Highway网络
            highway_out = 0
            if x.size(1) >= self.highway_window:
                highway_input = x[:, -self.highway_window:, 0]  # 只取第一个特征
                highway_out = self.highway(highway_input)

            # 组合输出
            output = main_out + skip_out + highway_out

            return output


class DeepLearningTrainer:
    """深度学习模型训练器"""

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"🔧 使用设备: {self.device}")

        self.scaler = MinMaxScaler()

    def prepare_sequences(self, data, target_col, feature_cols, seq_length=30, train_ratio=0.8):
        """准备序列数据"""
        # 选择特征和目标
        features = data[feature_cols].fillna(method='ffill').fillna(0)
        target = data[target_col].fillna(method='ffill').fillna(0)

        # 数据标准化
        features_scaled = self.scaler.fit_transform(features)
        target_scaled = target.values.reshape(-1, 1)

        # 创建序列
        X, y = [], []
        for i in range(seq_length, len(features_scaled)):
            X.append(features_scaled[i-seq_length:i])
            y.append(target_scaled[i])

        X, y = np.array(X), np.array(y)

        # 分割训练和测试集
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return (X_train, y_train), (X_test, y_test)

    def create_data_loaders(self, X_train, y_train, X_test, y_test, batch_size=32):
        """创建数据加载器"""
        # 转换为Tensor
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)

        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def train_model(self, model, train_loader, test_loader, epochs=100, lr=0.001, patience=10):
        """训练深度学习模型"""
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"🚀 开始训练 {model.__class__.__name__} 模型...")

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(test_loader)
            val_losses.append(avg_val_loss)

            # 学习率调度
            scheduler.step(avg_val_loss)

            # 早停
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), f'best_{model.__class__.__name__.lower()}.pth')
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

            if patience_counter >= patience:
                print(f"早停于第 {epoch} 轮，最佳验证损失: {best_val_loss:.6f}")
                break

        # 加载最佳模型
        model.load_state_dict(torch.load(f'best_{model.__class__.__name__.lower()}.pth'))

        print(f"✅ {model.__class__.__name__} 模型训练完成")

        return model, train_losses, val_losses

    def predict(self, model, X_test):
        """模型预测"""
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = model(X_test_tensor)
            return predictions.cpu().numpy()


class ModelPipeline:
    """模型管道 - 整合所有模型"""

    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.models = {}
        self.predictions = {}

    def run_sarimax(self, df, target_col='日收益率', exog_cols=None):
        """运行SARIMAX模型"""
        print("\n" + "="*60)
        print("🔮 SARIMAX 时间序列模型")
        print("="*60)

        sarimax = SARIMAXModel()
        y, exog = sarimax.prepare_data(df, target_col, exog_cols)

        if len(y) < 50:
            print("❌ 数据量不足，跳过SARIMAX模型")
            return None

        # 分割训练和测试集
        train_size = int(len(y) * 0.8)
        y_train = y[:train_size]
        y_test = y[train_size:]

        if exog is not None:
            exog_train = exog[:train_size]
            exog_test = exog[train_size:]
        else:
            exog_train = exog_test = None

        # 训练模型
        fitted_model = sarimax.fit(y_train, exog_train)
        if fitted_model is None:
            return None

        # 预测
        if exog_test is not None:
            predictions = fitted_model.forecast(steps=len(y_test), exog=exog_test)
        else:
            predictions = fitted_model.forecast(steps=len(y_test))

        # 评估
        metrics = self.evaluator.calculate_metrics(y_test, predictions, 'SARIMAX')
        self.evaluator.print_metrics('SARIMAX')

        # 保存结果
        self.models['SARIMAX'] = sarimax
        self.predictions['SARIMAX'] = {
            'y_test': y_test,
            'predictions': predictions,
            'model': fitted_model
        }

        return sarimax

    def run_deep_learning_models(self, df, target_col='日收益率', feature_cols=None, seq_length=30):
        """运行深度学习模型"""
        if not TORCH_AVAILABLE:
            print("❌ PyTorch未安装，跳过深度学习模型")
            return

        # 自动选择特征列
        if feature_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != target_col][:20]  # 限制特征数量

        print(f"\n使用特征: {feature_cols[:5]}...")

        trainer = DeepLearningTrainer()

        # 准备数据
        (X_train, y_train), (X_test, y_test) = trainer.prepare_sequences(
            df, target_col, feature_cols, seq_length
        )

        if len(X_train) < 50:
            print("❌ 数据量不足，跳过深度学习模型")
            return

        train_loader, test_loader = trainer.create_data_loaders(
            X_train, y_train, X_test, y_test
        )

        input_size = X_train.shape[2]

        # 运行各种深度学习模型
        models_to_train = [
            ('LSTM', LSTMModel(input_size)),
            ('Transformer', TransformerModel(input_size)),
            ('LSTNet', LSTNetModel(input_size))
        ]

        for model_name, model in models_to_train:
            print(f"\n" + "="*60)
            print(f"🧠 {model_name} 深度学习模型")
            print("="*60)

            try:
                # 训练模型
                trained_model, train_losses, val_losses = trainer.train_model(
                    model, train_loader, test_loader, epochs=50
                )

                # 预测
                predictions = trainer.predict(trained_model, X_test).flatten()
                y_test_flat = y_test.flatten()

                # 评估
                metrics = self.evaluator.calculate_metrics(y_test_flat, predictions, model_name)
                self.evaluator.print_metrics(model_name)

                # 保存结果
                self.models[model_name] = trained_model
                self.predictions[model_name] = {
                    'y_test': y_test_flat,
                    'predictions': predictions,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }

            except Exception as e:
                print(f"❌ {model_name} 模型训练失败: {e}")
                continue

    def visualize_results(self, save_path='reports/figures'):
        """可视化结果"""
        if not self.predictions:
            print("❌ 没有预测结果可供可视化")
            return

        os.makedirs(save_path, exist_ok=True)

        # 1. 预测结果对比图
        plt.figure(figsize=(15, 10))

        n_models = len(self.predictions)
        cols = 2
        rows = (n_models + 1) // 2

        for i, (model_name, results) in enumerate(self.predictions.items()):
            plt.subplot(rows, cols, i+1)

            y_test = results['y_test']
            predictions = results['predictions']

            # 只显示前100个点，避免图表过于拥挤
            n_points = min(100, len(y_test))

            plt.plot(y_test[:n_points], label='实际值', alpha=0.7)
            plt.plot(predictions[:n_points], label='预测值', alpha=0.7)
            plt.title(f'{model_name} 预测结果')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}/model_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. 模型性能比较
        comparison_df = self.evaluator.compare_models()

        if comparison_df is not None:
            plt.figure(figsize=(12, 6))

            # RMSE比较
            plt.subplot(1, 2, 1)
            comparison_df['RMSE'].plot(kind='bar')
            plt.title('模型RMSE比较')
            plt.ylabel('RMSE')
            plt.xticks(rotation=45)

            # R²比较
            plt.subplot(1, 2, 2)
            comparison_df['R²'].plot(kind='bar')
            plt.title('模型R²比较')
            plt.ylabel('R²')
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(f'{save_path}/model_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 3. 深度学习模型训练曲线
        dl_models = ['LSTM', 'Transformer', 'LSTNet']
        dl_results = {k: v for k, v in self.predictions.items() if k in dl_models}

        if dl_results:
            plt.figure(figsize=(15, 5))

            for i, (model_name, results) in enumerate(dl_results.items()):
                if 'train_losses' in results:
                    plt.subplot(1, len(dl_results), i+1)
                    plt.plot(results['train_losses'], label='训练损失')
                    plt.plot(results['val_losses'], label='验证损失')
                    plt.title(f'{model_name} 训练曲线')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{save_path}/training_curves.png', dpi=300, bbox_inches='tight')
            plt.show()

        print(f"✅ 可视化结果已保存到 {save_path}")

    def generate_report(self, save_path='reports'):
        """生成模型报告"""
        os.makedirs(save_path, exist_ok=True)

        report_content = f"""
# 股票预测模型分析报告

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 模型概述
本次分析使用了以下模型进行股票收益率预测：
- SARIMAX: 经典时间序列模型
- LSTM: 长短期记忆网络
- Transformer: 注意力机制模型  
- LSTNet: 长短期时间序列网络

## 模型性能比较
"""

        # 添加性能比较表
        if self.evaluator.metrics:
            comparison_df = pd.DataFrame(self.evaluator.metrics).T
            report_content += f"\n```\n{comparison_df.to_string()}\n```\n"

        report_content += f"""

## 模型详细分析

### 各模型特点
1. **SARIMAX**: 适合有明确季节性和趋势的时间序列
2. **LSTM**: 能够捕捉长期依赖关系，适合复杂的非线性模式
3. **Transformer**: 通过注意力机制关注重要的历史信息
4. **LSTNet**: 结合CNN和LSTM，同时处理短期和长期模式

### 建议
- 对于稳定的市场环境，推荐使用传统的SARIMAX模型
- 对于波动较大的市场，深度学习模型可能表现更好
- 建议使用集成方法结合多个模型的预测结果

## 风险提示
- 股票预测存在很大不确定性，模型结果仅供参考
- 过去的表现不能保证未来的收益
- 请结合其他分析方法和风险管理策略
"""

        # 保存报告
        report_path = f'{save_path}/predictive_models_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"✅ 模型报告已保存到 {report_path}")

        return report_path


def run_all_models(comprehensive_features, target_col='日收益率', feature_cols=None):
    """
    运行所有预测模型的主函数

    参数:
    - comprehensive_features: 综合特征数据
    - target_col: 目标列名
    - feature_cols: 特征列名列表
    """
    print("\n" + "🚀"*20)
    print("🎯 开始运行所有预测模型")
    print("🚀"*40)

    # 创建模型管道
    pipeline = ModelPipeline()

    # 数据预处理
    df = comprehensive_features.copy()

    # 移除缺失值过多的行
    df = df.dropna(thresh=len(df.columns)*0.7)

    if len(df) < 100:
        print("❌ 有效数据量不足100行，无法进行建模")
        return None

    print(f"📊 使用数据: {len(df)} 行, {len(df.columns)} 列")

    # 自动选择外生变量（SARIMAX用）
    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exog_cols = [col for col in numeric_cols if col != target_col and '年' not in col and '月' not in col][:10]
    else:
        exog_cols = feature_cols[:10]

    print(f"🔧 SARIMAX外生变量: {exog_cols[:3]}...")

    # 1. 运行SARIMAX模型
    try:
        pipeline.run_sarimax(df, target_col, exog_cols)
    except Exception as e:
        print(f"❌ SARIMAX运行失败: {e}")

    # 2. 运行深度学习模型
    try:
        pipeline.run_deep_learning_models(df, target_col, exog_cols)
    except Exception as e:
        print(f"❌ 深度学习模型运行失败: {e}")

    # 3. 生成可视化和报告
    try:
        pipeline.visualize_results()
        pipeline.generate_report()
    except Exception as e:
        print(f"❌ 生成报告失败: {e}")

    print("\n" + "🎉"*20)
    print("🏆 所有模型运行完成！")
    print("🎉"*40)

    return pipeline


if __name__ == "__main__":
    # 测试代码
    print("🧪 预测模型模块测试")
