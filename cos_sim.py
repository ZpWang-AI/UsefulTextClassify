import hanlp

from tqdm import tqdm

from config import CustomConfig
from corpus import *


def hanlp_tutorial():
    # !pip install hanlp -U
    # hanlp.pretrained.sts.ALL # 语种见名称最后一个字段或相应语料库
    sts = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)
    res = sts([
        ('看图猜一电影名', '看图猜电影'),
        ('无线路由器怎么无线上网', '无线上网卡和无线路由器怎么用'),
        ('北京到上海的动车票', '上海到北京的动车票'),
    ])
    print(res)
    
    
def main():
    config = CustomConfig()
    test_data_file = test_data_file_list[1]
    save_res_path = './data/result_sim_data2.xlsx'
    
    test_data = preprocess_test_data(test_data_file)
    test_data = CustomDataset(test_data, config, phase='test')
    test_data = DataLoader(test_data, batch_size=16, shuffle=False)
    
    sts = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)
    res = []
    for piece in tqdm(test_data):
        piece = list(zip(*piece))
        res_batch = sts(piece)
        res.extend(res_batch)
    
    test_data_content = pd.read_excel(test_data_file, sheet_name=1)
    test_data_content['non_answer (not marked)'] = res
    test_data_content.rename(columns={'non_answer (not marked)': 'similarity'}, inplace=True)
    writer = pd.ExcelWriter(save_res_path)
    test_data_content.to_excel(writer)
    writer.save()


if __name__ == '__main__':
    main()
    
    pass