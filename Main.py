#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:44
# @Author  : Tianyu Liu

import sys
import os
import tensorflow as tf
import time
from SeqUnit import *
from DataLoader import DataLoader
import numpy as np
from PythonROUGE import PythonROUGE
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from preprocess import *
from util import * 


tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 400, "Size of embedding.")
tf.app.flags.DEFINE_integer("field_size", 50, "Size of embedding.")
tf.app.flags.DEFINE_integer("pos_size", 5, "Size of embedding.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size of train set.")
tf.app.flags.DEFINE_integer("epoch", 50, "Number of training epoch.")                    ## 에폭 50
tf.app.flags.DEFINE_integer("source_vocab", 20003,'vocabulary size')
tf.app.flags.DEFINE_integer("field_vocab", 1480,'vocabulary size')
tf.app.flags.DEFINE_integer("position_vocab", 31,'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 20003,'vocabulary size')
tf.app.flags.DEFINE_integer("report", 5000,'report valid results after some steps')      ## 5000번에 한번 레포트
tf.app.flags.DEFINE_float("learning_rate", 0.0003,'learning rate')

tf.app.flags.DEFINE_string("mode",'train','train or test')                               ## 처음엔 train 모드
tf.app.flags.DEFINE_string("load",'0','load directory') # BBBBBESTOFAll                  ## (테스트 시 ) 모델 가져오기
tf.app.flags.DEFINE_string("dir",'processed_data','data set directory')                  ## processed_data폴더의 파일.
tf.app.flags.DEFINE_integer("limits", 0,'max data set size')


tf.app.flags.DEFINE_boolean("dual_attention", True,'dual attention layer or normal attention')
tf.app.flags.DEFINE_boolean("fgate_encoder", True,'add field gate in encoder lstm')

tf.app.flags.DEFINE_boolean("field", False,'concat field information to word embedding')
tf.app.flags.DEFINE_boolean("position", False,'concat position information to word embedding')
tf.app.flags.DEFINE_boolean("encoder_pos", True,'position information in field-gated encoder')
tf.app.flags.DEFINE_boolean("decoder_pos", True,'position information in dual attention decoder')


FLAGS = tf.app.flags.FLAGS
last_best = 0.0

gold_path_test = 'processed_data/test/test_split_for_rouge/gold_summary_'
gold_path_valid = 'processed_data/valid/valid_split_for_rouge/gold_summary_'

# test phase
if FLAGS.load != "0":
    save_dir = 'results/res/' + FLAGS.load + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = 'results/evaluation/' + FLAGS.load + '/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'
# train phase
else:
    prefix = str(int(time.time() * 1000))                     ## 160000
    save_dir = 'results/res/' + prefix + '/'                  ## save_dir = results/res/160000/
    save_file_dir = save_dir + 'files/'                       ## save_file_dir = results/res/160000/files/
    pred_dir = 'results/evaluation/' + prefix + '/'           ## pred_dir = results/evaluation/160000/
    os.mkdir(save_dir)                                        ## results/res/160000/ 디렉토리 생성
    if not os.path.exists(pred_dir):                          ## results/evaluation/160000/ 가 없으면 생성
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):                     ## results/res/160000/ 가 없으면 생성
        os.mkdir(save_file_dir)                               
    pred_path = pred_dir + 'pred_summary_'                   ## pred_path = results/evaluation/160000/pred_sumary_
    pred_beam_path = pred_dir + 'beam_summary_'              ## pred_beam_path = results/evaluation/160000/beam_summary_

log_file = save_dir + 'log.txt'                              ## log_file = results/res/160000/log.txt


def train(sess, dataloader, model):                                                           ## train 모드
    write_log("#######################################################")
    for flag in FLAGS.__flags:
        write_log(flag + " = " + str(FLAGS.__flags[flag]))                                         ## 상단에 정의한 각 FLAGS 적용 및 출력
    write_log("#######################################################")
    trainset = dataloader.train_set                                                                ## trainset = train_data_path[train.summary.id, train.box.val.id ......]
    k = 0
    loss, start_time = 0.0, time.time()
    for _ in range(FLAGS.epoch):                                                                   ## epoch[50]만큼 반복
        for x in dataloader.batch_iter(trainset, FLAGS.batch_size, True):                               ## trainset[582,659]을 배치 사이즈[32]만큼 shuffle해서 x에 집어 넣음 x는 18,209번 반복
            loss += model(x, sess)                                                                         ## loss = 32개의 trainset을 모델의 세션 실행해 loss 계산
            k += 1                                                                                         ## k + 1 [k 하나에 32개의 train data set이 들어감]
            progress_bar(k%FLAGS.report, FLAGS.report)                                                     ## 프로그레스 바 함수(현재[k], 전체[5000])
            if (k % FLAGS.report == 0):                                                                    ## k % 5000 == 0이면? 즉 k가 5000, 10000, 15000... 이면
                cost_time = time.time() - start_time                                                            ## 걸린 시간
                write_log("%d : loss = %.3f, time = %.3f " % (k // FLAGS.report, loss, cost_time))              ## step[1,2,3...]   // : k는 1~end까지. k//5000이 step. 
                loss, start_time = 0.0, time.time()                                                             ## loss 는 다시 0
                if k // FLAGS.report >= 1:                                                                      ## step 1부터~ step번째 모델 세이브. 
                    ksave_dir = save_model(model, save_dir, k // FLAGS.report)                             
                    write_log(evaluate(sess, dataloader, model, ksave_dir, 'valid'))
                    


def test(sess, dataloader, model):
    evaluate(sess, dataloader, model, save_dir, 'test')

def save_model(model, save_dir, cnt):                                    ## 모델 저장 save_model(모델, results/res/160000/, cnt)
    new_dir = save_dir + 'loads' + '/'                                      ## new_dir = results/res/160000/loads/
    if not os.path.exists(new_dir):                                         ## results/res/160000/loads/ 가 존재하지 않으면 디렉토리 생성
        os.mkdir(new_dir)                                                                                           ## cnt = 0, 1, 2, 3, ....
    nnew_dir = new_dir + str(cnt) + '/'                                     ## nnew_dir = results/res/160000/loads/cnt/
    if not os.path.exists(nnew_dir):                                        ## results/res/160000/loads/cnt/ 가 존재하지 않으면 디렉토리 생성
        os.mkdir(nnew_dir)
    model.save(nnew_dir)                                                    ## results/res/160000/loads/0..1..2..3../ 에 모델 저장
    return nnew_dir

def evaluate(sess, dataloader, model, ksave_dir, mode='valid'):          ## 평가 evaluate
    if mode == 'valid':                                                     ## valid 모드이면
        # texts_path = "original_data/valid.summary"
        texts_path = "processed_data/valid/valid.box.val"                      ## texts_path = processed_data/valid/valid.box.val
        gold_path = gold_path_valid                                            ## gold_path = gold_path_valid
        evalset = dataloader.dev_set                                           ## evalset = valid.summary.id , valid.box.val.id , valid.box.lab.id , valid.box.pos , valid.box.rpos
    else:                                                                   ## valid 모드 아니면
        # texts_path = "original_data/test.summary"
        texts_path = "processed_data/test/test.box.val"                        ## texts_path = gold_path_test
        gold_path = gold_path_test                                             ## gold_path = gold_path_test
        evalset = dataloader.test_set                                          ## evalset = test 셋 또는 train 셋
    
    # for copy words from the infoboxes
    texts = open(texts_path, 'r').read().strip().split('\n')                ## valid 기준 . processed_data/valid/valid.box.val의
    texts = [list(t.strip().split()) for t in texts]
    v = Vocab()

    # with copy
    pred_list, pred_list_copy, gold_list = [], [], []                      ## copy의 경우 pred_list, pred_list_copy, gold_list
    pred_unk, pred_mask = [], []
    
    k = 0
    for x in dataloader.batch_iter(evalset, FLAGS.batch_size, False):      ## valid 기준 valid.summary.id, valid.box.val.id, ...의 데이터를 배치사이즈 32로 shuffle 없이
        predictions, atts = model.generate(x, sess)                        ## model로 predictions와 atts 생성
        atts = np.squeeze(atts)                                            
        idx = 0
        for summary in np.array(predictions):                                  ## predictions 당 summary
            with open(pred_path + str(k), 'w') as sw:                              ## results/evaluation/160000/pred_sumary_k  0, 1, 2, ...
                summary = list(summary)
                if 2 in summary:
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                real_sum, unk_sum, mask_sum = [], [], []
                for tk, tid in enumerate(summary):
                    if tid == 3:
                        sub = texts[k][np.argmax(atts[tk,: len(texts[k]),idx])]
                        real_sum.append(sub)
                        mask_sum.append("**" + str(sub) + "**")
                    else:
                        real_sum.append(v.id2word(tid))
                        mask_sum.append(v.id2word(tid))
                    unk_sum.append(v.id2word(tid))
                sw.write(" ".join([str(x) for x in real_sum]) + '\n')
                pred_list.append([str(x) for x in real_sum])
                pred_unk.append([str(x) for x in unk_sum])
                pred_mask.append([str(x) for x in mask_sum])
                k += 1
                idx += 1
    write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
    write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")


    for tk in range(k):
        with open(gold_path + str(tk), 'r') as g:                                  ## 'processed_data/valid/valid_split_for_rouge/gold_summary_tk  0, 1, 2, 3, ...
            gold_list.append([g.read().strip().split()])

    gold_set = [[gold_path + str(i)] for i in range(k)]                          ## gold_set = processed_data/valid/valid_split_for_rouge/gold_summary_i  k번
    pred_set = [pred_path + str(i) for i in range(k)]                            ## pred_set = results/evaluation/160000/pred_sumary_i   k번

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_list)
    copy_result = "with copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
    (str(F_measure), str(recall), str(precision), str(bleu))
    # print copy_result

    for tk in range(k):
        with open(pred_path + str(tk), 'w') as sw:
            sw.write(" ".join(pred_unk[tk]) + '\n')

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_unk)
    nocopy_result = "without copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
    (str(F_measure), str(recall), str(precision), str(bleu))
    # print nocopy_result
    result = copy_result + nocopy_result 
    # print result
    if mode == 'valid':
        print result

    return result



def write_log(s):
    print s
    with open(log_file, 'a') as f:
        f.write(s+'\n')


def main():
    config = tf.ConfigProto(allow_soft_placement=True)                  ## 자동으로 gpu 할당
    config.gpu_options.allow_growth = True                              ## gpu 사용률 증가시키기
    with tf.Session(config=config) as sess:                             ## sess = 위의 옵션 사용
        copy_file(save_file_dir)                                        ## copy_file(save_file_dir = results/res/160000/files/)
        dataloader = DataLoader(FLAGS.dir, FLAGS.limits)                ## dataloader = DataLoader(processed_data, 0)
        model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size, emb_size=FLAGS.emb_size,
                        field_size=FLAGS.field_size, pos_size=FLAGS.pos_size, field_vocab=FLAGS.field_vocab,
                        source_vocab=FLAGS.source_vocab, position_vocab=FLAGS.position_vocab,                                ## 모델 생성(정보는 맨 위에)
                        target_vocab=FLAGS.target_vocab, scope_name="seq2seq", name="seq2seq",
                        field_concat=FLAGS.field, position_concat=FLAGS.position,
                        fgate_enc=FLAGS.fgate_encoder, dual_att=FLAGS.dual_attention, decoder_add_pos=FLAGS.decoder_pos,
                        encoder_add_pos=FLAGS.encoder_pos, learning_rate=FLAGS.learning_rate)
        sess.run(tf.global_variables_initializer())                     ## 위의 변수 할당 및 세션 시작
        # copy_file(save_file_dir)                                      ## copy_file(save_file_dir = results/res/160000/files/)
        if FLAGS.load != '0':
            model.load(save_dir)                                        ## load에 뭐 있으면 모델 로드(test시)
        if FLAGS.mode == 'train':
            train(sess, dataloader, model)                              ## train모드이면 train(모델[FLAGS]로 dataloader[processed_data, 0]의 세션[자동 gpu, gpu 증가] 시작
        else:
            test(sess, dataloader, model)                               ## test모드이면


if __name__=='__main__':
    # with tf.device('/gpu:' + FLAGS.gpu):
    main()
