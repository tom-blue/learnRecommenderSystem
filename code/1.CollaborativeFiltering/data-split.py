#-*-coding:utf-8-*-
'''
将ml-1m的rating.dat数据分割成训练集和测试集
'''


import operator

def round_int(rating_num, ratio):
    '''
    get the size of training data for each user
    Inputs:
        @rating_num: the total number of ratings for a specific user
        @ration: the percentage for training data
    Output:
        @train_size: the size of training data
    '''

    train_size = int(round(rating_num * ratio, 0))

    return train_size

def load_data(fr_rating):
    '''
    load the user-item rating data with the timestamp
    Input: 
        @fr_rating: the user-item rating data
    Output:
        @rating_data: user-specific rating data with timestamp
    '''

    rating_data = {}
    user_list = []
    for line in fr_rating:
        lines = line.split('::')
        user = lines[0]
        item = lines[1]
        rating = lines[2]
        time = lines[3].replace('\n', '')

        if user in rating_data:
            rating_data[user].append({item: rating})
        else:
            rating_data.update({user: [{item: rating}]})

        if user not in user_list:
            user_list.append(user)
    print(rating_data)
    return rating_data

def split_rating_into_train_test(rating_data, fw_train, fw_test, ratio):
    '''
    split rating_rating data into training and test data by timestamp
    Inputs:
        @rating_data: the user-specific rating data
        @fw_train: the training data file
        @fw_test: the test data file
        @ratio: the percentage of training data
    '''
    for user in rating_data:
        item_list = rating_data[user]
        rating_num = len(item_list)
        train_size = round_int(rating_num, ratio)

        flag = 0

        for item_rating in item_list:
            for key,value in item_rating.items():
                if flag < train_size:
                    line = user + '\t' + key +'\t' + value+ '\n'
                    fw_train.write(line)
                    flag = flag + 1
                else:
                    line = user + '\t' + key + '\t' + value + '\n'
                    fw_test.write(line)


if __name__ == '__main__':
    rating_file = '../../data/ml-1m/ratings.dat'
    train_file = './ml_train.txt'
    test_file = './ml_test.txt'

    fr_rating = open(rating_file, 'r')
    fw_train = open(train_file, 'w')
    fw_test = open(test_file, 'w')


    ratio = 0.8
    rating_data = load_data(fr_rating)
    split_rating_into_train_test(rating_data, fw_train, fw_test, ratio)

    fr_rating.close()
    fw_train.close()
    fw_test.close()