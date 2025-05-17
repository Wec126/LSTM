## 任务分析：

对数据进行预处理，构建模型使得模型可以回看336小时(14天)的数据并且预测未来24小时的数据；任务的预测目标是预测2017.07.01 之后的数据，并且采用每月滚动更新数据的训练形式：要求在2016.7 – 2017.6上训练模型，预测2017.7的数据，然后加入2017.7的数据到训练集合中再训练模型，依次类推

## 实验数据：

- 时间跨度： 7 – 2018.7
- 数据字段：
    - Date 例：2016/7/1 0:00
    - HUFL
    - HULL
    - MUFL
    - MULL
    - LUFL
    - LULL
    - OT：主要变量
    
    ![image](https://github.com/user-attachments/assets/5c12aa2d-1cc6-4b61-afad-fc9975952b9c)

    
    其中，OT的主要变化如上图所示；如上图所示，可以看到OT具有明显的周期性，例如在冬季(11月-2月期间，其OT含量明显较低)
    

## 实验算法

基于实验要求和任务要求，在本次实验中使用的模型为LSTM模型，并且结合数据处理操作等操作

### 1. 数据加载与预处理

- 使用pandas库加载ETTH1.csv数据集
- 使用2016年7月到2017年6月期间每个特征变量的均值($\mu$)和标准差($\sigma$)来进行z-score标准化：
    
    $$
    x' = (x-\mu)/ \sigma
    $$
    
    标准化后的数据是7维 $\{x’_t|x’_t \in \mathbb{R}^7\}_t$
    
- 滑动窗口处理：
    
    设置历史回看窗口长度 $T=336小时$
    
    设置预测窗口长度 $H = 24小时$
    
    构建训练和预测所需的滑动窗口数据对：利用前336小时的数据预测未来24小时的数据
    
    $$ 
    X[i] = \{X_{i-look\_back+1},...,X_i\}
    $$
    
    $$
    Y[i] = \{X_{i+1},..,X_{i+horizon}\}
    $$
    
    其中 $\text{look\_back}$为336， $\text{horizon}$为24
    

### 2.  模型构建

在本次实验中，为了实现可以从某个时间点向前看以及向后看，在这里使用了LSTM模型。

其中LSTM模型的结构为：

```python
self.lstm = nn.LSTM(input_dim,hidden_dim,
                    num_layers,batch_first = True,
                    dropout=dropout)
self.proj = nn.Linear(hidden_dim,horizon) 
```

### 3. 数据划分与滚动训练预测

初试训练集： 2016年7月1日 00:00:00到2017年6月30日 23:00:00的标准化数据

滚动预测流程：从2017年7月1日 00:00:00之后的数据开始，按月进行滚动：

- 在当前的训练集上训练模型
- 使用训练好的模型，回看前336小时的数据，预测未来24小时的数据，循环预测整个月份的数据
- 计算本月预测结果的评估指标(MSE 和 MAE)
- 将本月的真实数据加入到训练集中，更新训练集
- 在更新后的训练集上重新训练数据
- 重复以上步骤，预测下一个月的数据，直到预测完2018年7月的数据

下面为模型训练的超参数：

epoch：10

batch_size : 32

hidden_num : 128

dropout：0.2

优化器为：Adam，其中学习率参数为1e-3

使用MSELoss作为损失函数

并且在评测过程中使用MSE和MAE两个指标进行测试：

其中：

- MSE表示均方误差，用于计算预测值和真实值之间差值的平方的平均值
    
    MSE越小说明预测结果越接近真实值，模型的性能越好。
    
    此外，由于MSE是平方值，为了防止MSE过大，导致可视化过程中的指标无法精准显示，因此计算了RMSE，即MSE的开方。**如果RMSE若远大于MAE，则说明有少数时刻误差也别大(尖峰预测不好)**
    
    ![image](https://github.com/user-attachments/assets/34720fbb-09f4-4840-828a-a359ba1ecf09)

    
    [上述为使用MSE的可视化结果可以看到Pred和True的值无法完整展示]
    
- MAE表示平均绝对误差，用于计算预测值与真实值之间差值的绝对值的平均值。
    
    MAE越小说明预测结果越接近真实值，模型的性能越好。**当前OT在摄氏度的量纲下， $\text{MAE} < 1 \degree\text{C}$通常算是不错的效果； $\text{MAE} \approx 2-1 \degree \text{C}$则一般； $\text{MAE} > 5 \degree \text{C}$则说明模型拟合能力很弱**
    

![image](https://github.com/user-attachments/assets/af5b6d3b-e540-44ce-b28f-95c688945f98)


![image](https://github.com/user-attachments/assets/679a4bf3-bf4f-4446-8237-55ecaf65778d)


以上为训练的结果，可以看到几个现象：

1. **True Mean呈现显著地季节性-趋势变化**：
    
    其中12月份最低，然后又在春季回升，体现了典型的气温年周期特征
    
2. **Pred Mean整体波动很小**：
    
    说明模型有较强的均值回归，但是没有跟上真实曲线的整体走势
    
3. MSE与MAE值较大：
    
    其中2017年12月份的MSE达到289，MAE达到16，并且其他月份的MAE值都大于6，说明当前模型的拟合能力较弱。
    

为了解决这种无法跟随季节性的问题，在原本模型的基础上添加了**偏置项**，用于鼓励预测序列的整体平均值更接近真实序列的平均值，从而减少季节性或趋势上的系统性偏差

**偏置项**(bias)的数学定义：

$$
\mathcal{L} = \text{MSE} + \lambda(\frac{1}{T}\sum_t(y_t - \hat{y}_t))^2 
$$

上述Loss主要在优化两个情况：

- 点对点的平方误差：让模型拟合短期的波动细节
- 每条预测序列的平均偏差：让模型对每个预测窗口的整体水平保持中立，不整体高估或者低谷

冬天(例如12月份)的真实值都比较低，而模型总是预测接近0，那么她的平均偏差 $b = \frac{1}{H}\sum(y_t - \hat{y}_t)$会是一个负数，表明模型整体预测偏高；模型通过添加偏置项可以会减少 $y与\hat{y}$之间的误差，从而努力让整个预测段的均值贴近真实段的均值

![image](https://github.com/user-attachments/assets/54817f33-a0d7-4d87-b312-2ef314c4fe94)


当前 $\lambda_{bias}$为0.1，可以看到与未使用偏置项的相比，MSE和MAE有一定的进步，其中MAE有明显的减少，并且MAE与RMSE之间的拟合度更好。但是月份的均值还是存在无法对齐的情况

为此，我们进行了一下改进：

根据对 $\lambda_{bias}$的实验我们发现，当 $\lambda_{bias}=1$时可以得到较好的评估结果，具体实验结果在附录

- 加入**时间特征**
    
    添加偏置项只是为了使得在一个局部时间内两个线段之间的均值可以趋向一致，但是依旧是一个局部的预测，但仍然无法捕捉长期的季节性趋势。
    
    LSTM无法通过当前的局部数据分析到当前是什么季节(因为在冬季可以明显看到真实均值的下降)模型只能学到短期局部模式，无法识别长期趋势
    
    为此，添加上一定的时间特征，可以让模型知道当前是几月/几点
    
    对已知数据集进行研究，最终确认了一下集合特征：
    
    1. hour_sin & hour_cos：可以用于捕捉日内周期
        
        ![image](https://github.com/user-attachments/assets/80c7cacc-54f1-4239-8c82-02d860d50bc8)


        
        并且根据真实数据发现，对于一天内的时间，白天时OT值要普遍大于夜晚，因此为了刚好预测，将时间加入到判断当中
        
    2. month_sin & month_cos：可以学习到季节性
    
    下述为在模型上对数据集进行处理，添加上时间特征之后的结果：
    
    ![image](https://github.com/user-attachments/assets/b4c05ee5-1a10-4e00-8071-2bc40522dd0a)


    
    ![image](https://github.com/user-attachments/assets/2bea110b-c85e-487e-9901-548b3d020bb3)

    
    根据上述可以看到MSE和MAE有明显改进，但是依旧可以从均值拟合程度上看到，真实均值和预期均值还是存在一定差距。这说明时间特征虽然增加了模型对周期性变化的感知能力，但是模型仍难以完全复刻真实的季节性趋势。
    
- 加入STL
    
    为了进一步提升模型对趋势变化的建模能力，我们引入STL(Seasonal-Trend decomposition using Losses分解技术)
    
    通过将原始时间序列拆分为：
    
    - 长期趋势项(trend)
    - 季节项(seasonal)
    - 残差项(residual)
    
    我们可以将建模任务聚焦在更容易拟合的**残差序列**上，同时在预测阶段将模型输出加回趋势与季节项，得到更合理的整体预测。
    
    ![image](https://github.com/user-attachments/assets/59023728-e3ca-4e73-82c9-c46e09a57c68)

    
    ![image](https://github.com/user-attachments/assets/1222035e-e340-4dbd-8785-7b9eff951781)

    
    上述为添加了STL之后的结果，可以明显看到MSE和MAE下降，并且True Mean和Pred Mean趋向于拟合(当前 $\lambda_{bias}=0.5$ ，更多数据请看appendix)
    
    ## Appendix
    
    ### 使用时间特征
    
    $\lambda_{bias}$ = 0.5
    
    ![image](https://github.com/user-attachments/assets/f8a4e82c-7db1-4d94-a9f9-d81e73a5740e)

    
    ![image](https://github.com/user-attachments/assets/fc14807c-5519-4c89-8178-dad19ba8a854)

    
    $\lambda_{bias}$ = 1.5
    
    ![image](https://github.com/user-attachments/assets/fed2b326-3367-4280-8073-2d2aa7eef524)

    
    ![image](https://github.com/user-attachments/assets/1ff5d602-e934-4e5d-8594-0e30b4365b96)

    
    ### 使用STL
    
    $\lambda_{bias} = 1.0$
    
    ![image](https://github.com/user-attachments/assets/201b622c-106d-4603-ae8a-a9c444f5f804)

    
    ![image](https://github.com/user-attachments/assets/c7d6cf25-56ad-4eb2-a9e2-3d0d17e96b6a)
