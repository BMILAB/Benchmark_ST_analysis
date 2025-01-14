
import scanpy as sc
from SpaceFlow import SpaceFlow
from utils import mk_dir
import os
import pandas as pd
def load_ST_file(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True, file_Adj=None):
    adata_h5 = sc.read_visium(file_fold, load_images=load_images, count_file=count_file)
    adata_h5.var_names_make_unique()

    if load_images is False:
        if file_Adj is None:
            file_Adj = os.path.join(file_fold, "spatial/tissue_positions_list.csv")

        positions = pd.read_csv(file_Adj, header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata_h5.obs = adata_h5.obs.join(positions, how="left")
        adata_h5.obsm['spatial'] = adata_h5.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
        adata_h5.obs.drop(columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)

    print('adata: (' + str(adata_h5.shape[0]) + ', ' + str(adata_h5.shape[1]) + ')')
    return adata_h5


import sys
sys.path.append('../')
import utils_for_all as usa
if __name__ == '__main__':

    results = pd.DataFrame()
    dataset1 = ["Stereo","Breast_cancer", "Mouse_brain", "STARmap","SeqFish","STARmap"]
    Dataset_test = ['151673']

for dataset in  Dataset_test:
    print(f"====================begin test on {dataset}======================================")

    if dataset.startswith('15'):
        save_path = f'../../Output/SpaceFlow/DLPFC/{dataset}/'
    else:
        save_path = f'../../Output/SpaceFlow/{dataset}/'
    mk_dir(save_path)

    results = pd.DataFrame()
    import psutil, time, tracemalloc
    for i in range(1):
        num = i + 1
        print("===Training epoch:{}====".format(num))
        start = time.time()
        tracemalloc.start()
        start_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
        adata_h5,n_cluster = usa.get_adata(dataset, data_path='../../Dataset/')

        sf = SpaceFlow.SpaceFlow(count_matrix=adata_h5.X,
                                 spatial_locs=adata_h5.obsm['spatial'])

        sf.preprocessing_data()
        sf.train(embedding_save_filepath=save_path,epochs=1000)

        print("train finished，begin segment！")
        res=sf.segmentation_1(adata_h5, dataset, save_path)

        end = time.time()
        end_MB = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
        uesd_time = end - start
        used_memo = end_MB - start_MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = peak / 1024.0 / 1024.0 / 1024.0
        print(u'Current memory usage_end:：%.4f GB' % used_memo)
        print('time: {:.4f} s'.format(uesd_time))
        print('memory blocks peak:{:>10.4f} GB'.format(peak))
        tracemalloc.clear_traces()
        res["time"] = uesd_time
        res["Memo"] = used_memo
        res["Memo_peak"] = peak
        res["round"] = i + 1
        results = results._append(res, ignore_index=True)

print(results.head())
results.to_csv(f'{save_path}result.csv')
results.set_index('dataset', inplace=True)

res_mean = results.mean()
res_mean.to_csv(f'{save_path}{dataset}_mean.csv', header=True)
res_std = results.std()
res_std.to_csv(f'{save_path}{dataset}_std.csv', header=True)
res_median = results.median()
res_median.to_csv(f'{save_path}{dataset}_median.csv', header=True)

