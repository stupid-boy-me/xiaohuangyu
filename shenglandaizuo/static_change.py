'''将CSV中的数据变成gmm.data中的数据格式'''
'''
ID	A	B	C                        0          0    
1	14964	1962	21762         0000000      0  0                 14964	1962	21762
2	6830.8	1299	40013            0        0    0                6830.8	1299	40013
3	18003.6	7194	156596           0000      0  0                 18003.6	7194	156596
                                                 0
'''
import pandas as pd

csv_file = r'C:\Users\Desktop\AI_traffic\shenglandaizuo\GMMdata.csv'
csv_data = pd.read_csv(csv_file, low_memory = False)#防止弹出警告
dat=csv_data.drop(['ID'],axis=1) #去除‘id’列
dat.drop([1]) # #删除1行的整行数据
csv_df = pd.DataFrame(dat)
csv_df.to_csv(r'C:\Users\Desktop\AI_traffic\shenglandaizuo\GMMdata_kongge.csv', sep=" ", header=False, index=False)


