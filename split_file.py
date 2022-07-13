import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parts', '-p', type=int, required=True)
    parser.add_argument('--file', '-f', type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--label', type=str, default='id')

    args = parser.parse_args()

    rf = open(args.file+".tsv", 'r')
    ori_data = rf.readlines()
    rf.close()
    rf2 = open(args.file+"."+args.label, 'r')
    ids = rf2.readlines()
    rf2.close()

    data_path = ori_data[0]

    new_data = [[] for _ in range(args.parts)]
    new_ids = [[] for _ in range(args.parts)]

    for i, j in enumerate(ori_data[1:]):
        new_data[i % args.parts].append(j)

    for i, j in enumerate(ids):
        new_ids[i % args.parts].append(j)

    file_name_prefix = args.file.split('/')[-1].split('.')[0]
    for i, new_j in enumerate(new_data):
        file_name = file_name_prefix + '_{}'.format(i)
        with open(os.path.join(args.datadir, file_name+'.tsv'), 'w') as f:
            f.write(data_path)
            f.writelines(new_j)
    for i, new_j in enumerate(new_ids):
        file_name = file_name_prefix + '_{}'.format(i)
        with open(os.path.join(args.datadir, file_name+'.'+args.label), 'w') as f:
            f.writelines(new_j)

