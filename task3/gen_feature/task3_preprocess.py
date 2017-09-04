#coding=utf-8
import numpy as np
import math
from boto.dynamodb.condition import NULL

base_path = 'E:/SMPCUP2017/'
task3_train_path = base_path + 'SMPCUP2017_task3_trainset/SMPCUP2017_TrainingData_Task3.txt'
uid_path = base_path + 'temp/all_uid.txt'
did_val_path = base_path + 'temp/all_did_val.txt'
input_path_full = base_path + 'input/full/'
input_path_split = base_path + 'input/split/'

u_post = dict()
u_browse = dict()
u_comment = dict()
u_vote_up = dict()
u_vote_down = dict()
u_favorite = dict()
u_follow = dict()
u_followed = dict()
u_letter = dict()
u_lettered = dict()
u_posts_val = dict()
u_best_post_val = dict()
u_worst_post_val = dict()

u_grow = dict()

uid_list_name = ['post', 'browse', 'comment', 'vote_up', 'vote_down', 'favorite', 'letter', 'lettered']

blog_list_name = ['browsed', 'commented', 'voted_up', 'voted_down', 'favorited']
d_record = dict()

            
def write_uid():
    with open(uid_path, 'w')as uid:
        for i in range(1,157248):
            name = 'U'
            for j in range(7-len(str(i))):
                name += '0' 
            name += str(i)
            uid.write(name + '\n')
            
def blog_val_write():
    with open(did_val_path, 'w')as did:
        for i in range(1,1000001):  #1000001
            name = 'D'
            for j in range(7-len(str(i))):
                name += '0' 
            name += str(i)
            
            weight = [1.0, 3.0, 2.0, -0.5, 4.0]
            w_sum = 0
            for i in weight:
                w_sum+=i
            
            if name not in d_record:
                did_val_1 = [0, 0, 0, 0, 0]
            else:
                did_val_1 = d_record[name]
            did_val = [math.sqrt(x) for x in did_val_1]
            did_val = [x*y for x, y in zip(weight, did_val)]
            val = 0
            for i in did_val:
                val +=i
            val = val/w_sum
            did.write(name + '\t' + str(val) + '\t' + ' '.join([str(x) for x in did_val_1]) + '\n')
            
            

def month_count(file, record, lettered = False, type='unknown'):
    if type in blog_list_name:
        type_num = blog_list_name.index(type)
    else:
        type_num = -1
        print 'post must be last one to month_count'
    lines = file.readlines()
    for line in lines:
        uid, did, date = line.strip().split('\001')
        if '-' not in date:
            year = date.strip()[:4]
            month = date.strip()[4:6]
            day = date.strip()[6:8]
        else:
            year, month, day = date.strip().split(' ')[0].split('-')
        
        if int(year)!=2015:
            print 'wrong year'
            break
        if int(month)>12 or int(month)<1:
            print 'wrong month'
            break
        
        if not lettered:
            if uid not in record:
                record[uid] = [0,0,0,0,0,0,0,0,0,0,0,0]
            record[uid][int(month)-1] = record[uid][int(month)-1] + 1
        else:
            if did not in record:
                record[did] = [0,0,0,0,0,0,0,0,0,0,0,0]
            record[did][int(month)-1] = record[did][int(month)-1] + 1
        
        if did not in d_record:
            d_record[did] = [0,0,0,0,0]
        if type_num>=0:
            d_record[did][type_num] = d_record[did][type_num] + 1
            
        if type == 'post':
            if uid not in u_posts_val:
                u_posts_val[uid] = []
                u_best_post_val[uid] = []
                u_worst_post_val[uid] = []
                for i in range(12):
                    u_posts_val[uid].append([0,0,0,0,0])
                    u_best_post_val[uid].append([0,0,0,0,0])
                    u_worst_post_val[uid].append([0,0,0,0,0])
            for i in range(len(blog_list_name)):
                u_posts_val[uid][int(month)-1][i] = u_posts_val[uid][int(month)-1][i] + d_record[did][i]
                if d_record[did][i]>u_best_post_val[uid][int(month)-1][i]:
                    u_best_post_val[uid][int(month)-1][i] = d_record[did][i]
                if d_record[did][i]<u_worst_post_val[uid][int(month)-1][i]:
                    u_worst_post_val[uid][int(month)-1][i] = d_record[did][i]
                
        
def write_tf_feature(path_x, path_y, path_list, path_x2):
    featrue = open(path_x, 'w')
    featrue2 = open(path_x2, 'w')
    tag = open(path_y, 'w')
    with open(path_list, 'r') as task3_train:
        lines = task3_train.readlines()
        for line in lines:
            if '\001' in line:
                uid, grow = line.strip().split('\001')
                tag.write(grow + '\n')
            else:
                uid = line.strip()
            if ' ' in uid:
                uid = uid.strip().replace(' ','')
            print(uid) 
            feature2_list =[0 for i in range(23)]
            uid_post_num_year = 0
            
            for i in range(12):
                list = []
                if uid in u_post:
                    list.append(str(u_post[uid][i]))
                else:
                    list.append('0')
                if uid in u_browse:
                    list.append(str(u_browse[uid][i]))
                else:
                    list.append('0')
                if uid in u_comment:
                    list.append(str(u_comment[uid][i]))
                else:
                    list.append('0')
                if uid in u_vote_up:
                    list.append(str(u_vote_up[uid][i]))
                else:
                    list.append('0')
                if uid in u_vote_down:
                    list.append(str(u_vote_down[uid][i]))
                else:
                    list.append('0')
                if uid in u_favorite:
                    list.append(str(u_favorite[uid][i]))
                else:
                    list.append('0')
                if uid in u_letter:
                    list.append(str(u_letter[uid][i]))
                else:
                    list.append('0')
                if uid in u_lettered:
                    list.append(str(u_lettered[uid][i]))
                else:
                    list.append('0')
#                 if uid in u_followed:
#                     list.append(str(u_followed[uid]))
#                 else:
#                     list.append('0')
#                 if uid in u_follow:
#                     list.append(str(u_follow[uid]))
#                 else:
#                     list.append('0')
                
                featrue_i = ' '.join(list)
                featrue.write(featrue_i)
                featrue.write(' ')
                if uid not in u_posts_val:
                    featrue_i_post_val = '0 0 0 0 0 0 0 0 0 0'
                else:
                    featrue_i_post_val = ' '.join([str(a) + ' ' + str(float(a)/(u_post[uid][i]+0.01)) for a in u_posts_val[uid][i]])
                featrue.write(featrue_i_post_val)
                featrue.write(' ')
                if uid not in u_best_post_val:
                    featrue_i_best_post = '0 0 0 0 0'
                else:
                    featrue_i_best_post = ' '.join([str(a) for a in u_best_post_val[uid][i]])
                featrue.write(featrue_i_best_post)
#                 featrue.write(' ')
#                 if uid not in u_worst_post_val:
#                     featrue_i_worst_post = '0 0 0 0 0'
#                 else:
#                     featrue_i_worst_post = ' '.join([str(a) for a in u_worst_post_val[uid][i]])
#                 featrue.write(featrue_i_worst_post)
                if i!=11:
                    featrue.write('\t')
                
                #begin compute the feature2
                for j in range(len(list)):
                    feature2_list[j]+=int(list[j])
                if uid in u_post:
                    uid_post_num_year+=u_post[uid][i]
                if uid in u_posts_val:
                    for j in range(len(blog_list_name)):
                        feature2_list[j+len(list)]+=u_posts_val[uid][i][j]
                        if feature2_list[j+len(list)+len(blog_list_name)]< u_best_post_val[uid][i][j]:
                            feature2_list[j+len(list)+len(blog_list_name)]= u_best_post_val[uid][i][j]   
            featrue.write('\n')
            
            for i in range(len(blog_list_name)):
                feature2_list[i+18] = float(feature2_list[i+8])/(uid_post_num_year+0.001)    
            featrue2_add = ' '.join([str(a) for a in feature2_list])
            featrue2.write(str(u_follow.get(uid, 0)) + ' ' + str(u_followed.get(uid, 0))+ ' ' + featrue2_add +'\n')
                    
                
    featrue.close()
    featrue2.close()
    tag.close()
    
    
def split_data(path, num):
    lines = NULL
    with open(path, 'r')as read:
        lines = read.readlines()
    path = path.replace('full', 'split')
    with open(path, 'w')as w:
        for line in lines[:num]:
            w.write(line)
    path_split = path.replace('.txt', '_split.txt')
    print path_split
    with open(path_split, 'w')as w:
        for line in lines[num:]:
            w.write(line)    
        
if __name__ == '__main__':
    with open(base_path + '8_Follow.txt', 'r') as follow:
        lines = follow.readlines()
        for line in lines:
            uid, uided = line.strip().split('\001')
            u_follow[uid] = u_follow.get(uid, 0) + 1
            u_followed[uided] = u_followed.get(uided, 0) + 1
     
    with open(base_path + '3_Browse.txt', 'r') as browse:
        month_count(browse, u_browse, type='browsed')
    with open(base_path+ '4_Comment.txt', 'r') as comment:
        month_count(comment, u_comment, type='commented')
    with open(base_path + '5_Vote-up.txt', 'r') as vote_up:
        month_count(vote_up, u_vote_up, type='voted_up')
    with open(base_path + '6_Vote-down.txt', 'r') as vote_down:
        month_count(vote_down, u_vote_down, type='voted_down')
    with open(base_path + '7_Favorite.txt', 'r') as favorite:
        month_count(favorite, u_favorite, type='favorited')
    #there is no follow
    with open(base_path + '9_Letter.txt', 'r') as letter:
        month_count(letter, u_letter)
        month_count(letter, u_lettered, lettered=True)
         
    with open(base_path+'2_Post.txt', 'r') as post:
        month_count(post, u_post, type='post')
    
    for uid in u_post:
        print uid, u_post[uid]
        
    print u_post['U0003142']
    
#     blog_val_write()
    
    path_x = input_path_full + 'task3_train_feature.txt'
    path_y = input_path_full + 'task3_train_tag.txt'
    path_x2 = input_path_full + 'task3_train_feature2.txt'
    write_tf_feature(path_x, path_y, task3_train_path, path_x2)
     
#     split_data(path_x, 900)
#     split_data(path_y, 900)
#     split_data(path_x2, 900)
         
    path_x = input_path_full + 'task3_validation_feature.txt'
    path_y = input_path_full + 'task3_validation_tag.txt'
    path_x2 = input_path_full + 'task3_validation_feature2.txt'
    path_list = base_path + 'SMPCUP2017_validation_set/SMPCUP2017_ValidationSet_Task3.txt'
    write_tf_feature(path_x, path_y, path_list, path_x2)
    
    path_x = input_path_full + 'task3_test_feature.txt'
    path_y = input_path_full + 'task3_test_tag.txt'
    path_x2 = input_path_full + 'task3_test_feature2.txt'
    path_list = base_path + 'SMPCUP2017_test_set/SMPCUP2017_TestSet_Task3.txt'
    write_tf_feature(path_x, path_y, path_list, path_x2)
    
    

        
    print 'code end'


    