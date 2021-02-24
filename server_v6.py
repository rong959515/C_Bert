from flask import Flask, request, abort, current_app
import pandas as pd
import numpy as np
from openpyxl import Workbook
import datetime
import logging
import re
from flask import Flask, request, render_template, jsonify, abort
from demo_v2 import Ans_setup
debug_userid = "U2a9439ce958eb77b60d6058b1bd8dc8c"
excel_file = 'QA_dataset_v9_20210218_975.xlsx'
df = pd.read_excel(excel_file, header=0,engine='openpyxl')
questions = df['question']
answers = df['answer']


def response(usr_qu):
    Model = Ans_setup(6,4,0.7)
    loc,prob,ans = Model.Ans(usr_qu)
    return loc,prob,ans

# ------------------ main program --------------------------
app = Flask(__name__)

question0  = '資訊管理學系特色是什麼？'

param0 = {'question' : question0,
          'answer' : 'result area'
        }

@app.route('/')
def forms():
  return render_template('tl_json_v4.html', param=param0)

# @app.route("/api/qa_v4", methods=['POST'])
# def api_qa_v4():
#     global builtin_qa, Qu
#
#     question = request.json['question']
#     userid = -1
#
#
#     qid, sim, answer = response(question)
#
#     if qid >=0:
#         QID = qid
#         DEP = df['department'][pid]
#         builtin_question = df['question'][pid]
#     else:
#         QID = qid
#         DEP = ''
#         builtin_question = ''
#
#
#     result_json = {
#         'question': question,
#         'answer': answer,
#         'qid': QID,
#         'department': DEP,
#         'similarity': sim,
#         'builtin_question': builtin_question
#     }
#     # print('result_json={}'.format(result_json))
#
#     return jsonify(result_json)

# 監聽所有來自 /callback 的 Post Request
# Input: { 'userid': 'Line ID', 'question': 'user question to linebot'}
# Output: 'answer string'
@app.route("/qa", methods=['POST'])
def api_qa():
    global app, logger, debug_userid

    remote_ip = request.remote_addr
    question = request.json['question']
    userid = request.json['userid']

    print('@' * 8, question)

    qid, sim, answer = response(question)

    if qid >=0:
        QID = qid
        DEP = df['department'][qid]
        builtin_question = df['question'][qid]
    else:
        QID = qid
        DEP = ''
        builtin_question = ''

    time = datetime.datetime.now()
    info_str = f' {remote_ip} - userid={userid}, qid={QID}, dep={DEP}, question=<{question}>, answer=<{answer}>'
    logger.info(info_str)


    clean_question = re.sub(r'[\x00-\x1F]+', ' ', question, flags=re.U)
    clean_answer   = re.sub(r'[\x00-\x1F]+', ' ', answer, flags=re.U)

    csv_str = f'{time} , {remote_ip} , {userid}, {QID} , {DEP} , {clean_question} , {clean_answer}'
    csv_logger.info(csv_str)
    # print(info_str)

    #ws.append([time, remote_ip, QID, DEP, question, answer])

    # Python 类型会被自动转换
    #wb.save("sample.xlsx")
    if userid == debug_userid:
       debug = f'[userid:{userid}, qid:{QID}, dep:{DEP}, bi_ques:{builtin_question}, sim:{sim}]'
       answer = f'{debug}\n{answer}'

    return f'{answer}'


import os
import sys

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.NOTSET)
    logger = logging.getLogger(__name__)

    handler = logging.FileHandler('server.log', encoding='UTF-8')
    handler.setLevel(logging.NOTSET)
    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
    handler.setFormatter(logging_format)
    logger.addHandler(handler)

    # sh = logging.StreamHandler(sys.stderr)
    # sh.setLevel(logging.NOTSET)
    # sh.setFormatter(logging_format)
    # logger.addHandler(sh)

    csv_logger = logging.getLogger('csv_logger')
    csv_handler = logging.FileHandler('server.csv', encoding='UTF-8')
    csv_handler.setLevel(logging.INFO)
    csv_logging_format = logging.Formatter('%(message)s')
    csv_handler.setFormatter(csv_logging_format)
    csv_logger.addHandler(csv_handler)


    #logger.debug('Debug message, should only appear in the file.')
    logger.info('Info message, should appear in file and stdout.')
    #logger.warning('Warning message, should appear in file and stdout.')
    #logger.error('Error message, should appear in file and stdout.')

    port = int(os.environ.get('PORT', 5009))
    app.debug = True
    app.run(threaded = True,host='0.0.0.0', port=port)