from flask import Blueprint, url_for, render_template, flash, request, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import redirect

from pibu import db
from pibu.forms import SocialLoginForm, UserLoginForm
from pibu.models import SocialLogin

bp = Blueprint('uvmuhwan', __name__, url_prefix='/uvmuhwan')

@bp.route('/list/')
def _list():
    page = request.args.get('page', type=int, default=1)  # 페이지
    social_login_list = SocialLogin.query.order_by(SocialLogin.id.desc())
    social_login_list = social_login_list.paginate(page=page, per_page=10)

    form = UserLoginForm()
    if request.method == 'POST' and form.validate_on_submit():
        error = None
        user = SocialLogin.query.filter_by(name=form.name.data).first()
        if not user:
            error = "존재하지 않는 사용자입니다."
        elif not check_password_hash(user.password, form.password.data):
            error = "전화번호가 올바르지 않습니다."
        if error is None:
            session.clear()
            session['user_id'] = user.id
            return redirect(url_for('main.index'))
        flash(error)

    return render_template('social_login_list.html', social_login_list=social_login_list, form=form)


@bp.route('/signup', methods=('GET', 'POST'))
def signup():
    form = SocialLoginForm()
    if request.method == 'POST' and form.validate_on_submit():
        user = SocialLogin.query.filter_by(name=form.name.data).first()
        if not user:
            user = SocialLogin(name=form.name.data,
                               password=form.password.data,
                               email=form.email.data,
                               gender=form.gender.data,
                               age=form.age.data,
                               agree_personal=form.agree_personal.data,
                               agree_location=form.agree_location.data
                               )
            db.session.add(user)
            db.session.commit()
            return redirect(url_for('main.index'))
        else:
            flash('이미 존재하는 사용자입니다.')
    return render_template('signup.html', form=form)

@bp.route('/login/', methods=('GET', 'POST'))
def login():
    form = UserLoginForm()
    if request.method == 'POST' and form.validate_on_submit():
        error = None
        user = SocialLogin.query.filter_by(name=form.name.data).first()
        if not user:
            error = "존재하지 않는 사용자입니다."
        elif user.password != form.password.data:
            error = "비밀번호가 올바르지 않습니다."
        if error is None:
            session.clear()
            session['user_id'] = user.id
            return redirect(url_for('main.index'))
        flash(error)
    return render_template('login.html', form=form)

@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = SocialLogin.query.get(user_id)

@bp.route('/logout/')
def logout():
    session.clear()
    return redirect(url_for('main.index'))

# 피부암 감지

import os
import numpy as np
from flask import Flask, redirect
from werkzeug.utils import secure_filename
from tensorflow import keras
import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert inputted image to an array
def process_image(file_path):
    print(file_path)
    image_array = []
    image_prep = keras.preprocessing.image.load_img(file_path, target_size=(224, 224, 3))
    image_prep = keras.preprocessing.image.img_to_array(image_prep)
    image_prep = preprocess_input(image_prep)
    image_array.append(image_prep)
    return np.array(image_array)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

def get_class(row):
	for c in row:
		if row[c] == 1:
			return c

# Load model and make prediction
def get_class_prediction(image_array):
    base_model = tensorflow.keras.applications.mobilenet.MobileNet()
    x = base_model.layers[-5].output
    x = Dropout(0.25)(x)
    predictions = Dense(7, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])
    model.load_weights('pibu/model/add_model.h5')
    classes = {
        0 : '광선각화증, 제자리암(Actinic keratoses/Bowens diease) - 양성이지만 악성이 될 수도 있다',
        1 : '기저세포암(Basal cell carcinoma) - 악성',
        2 : '양성태선각화증(Benign keratosis like lesions) - 양성',
        3 : '피부섬유종(Dermatofibroma) - 양성',
        4 : '악성흑색종(Melanoma) - 악성',
        5 : '점(Melanocytic nevi) - 양성',
		6 : '혈관피부병변(Vascular lesions) - 대부분 양성'
    }
    class_index = model.predict(image_array)
    class_re = np.argmax(class_index[0])
    print(class_index[0])
    print(class_re)
    return classes[class_re]

@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['file']
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            image_name = f.filename
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'static', secure_filename(f.filename))
            f.save(file_path)

            image = process_image(file_path)
            class_name = get_class_prediction(image).capitalize()
            return render_template('upload.html', label = class_name, img = image_name)
    return

#uvmap

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import folium
from datetime import datetime

excel_file_path = 'pibu/static/location_grids.xlsx'
excel_data = pd.read_excel(excel_file_path)

# 사용자가 입력한 지역명에 대응하는 행정구역 코드를 찾는 함수
def get_admin_code(location):
    for index, row in excel_data.iterrows():
        # 1단계, 2단계, 3단계 각 열에서 유사한 지역명을 찾음
        for level in ['1단계', '2단계', '3단계']:
            if not pd.isnull(row[level]) and location in row[level]:
                return row['행정구역코드']
    return None


def get_decimal_coordinates(admin_code):
    for index, row in excel_data.iterrows():
        if row['행정구역코드'] == admin_code:  # 행정구역 코드가 일치하는 경우에만 처리
            # 엑셀에서 각 격자의 위도와. 경도를 십진법으로 변환(지도에 나타내기 위함)
            decimal_longitude = row['경도(시)'] + (row['경도(분)'] / 60) + (row['경도(초)'] / 3600) + (
                    row['경도(초/100)'] / 360000)
            decimal_latitude = row['위도(시)'] + (row['위도(분)'] / 60) + (row['위도(초)'] / 3600) + (
                    row['위도(초/100)'] / 360000)
            return decimal_longitude, decimal_latitude
    return None, None


# 현재 자외선 지수를 정수로 변환하여 비교하도록 수정
def get_uv_warning_level(uv_index):
    if 0 <= uv_index < 3:
        level = '낮음'
    elif 3 <= uv_index < 6:
        level = '보통'
    elif 6 <= uv_index < 8:
        level = '높음'
    elif 8 <= uv_index < 11:
        level = '매우 높음'
    else:
        level = '위험'
    return level


def get_weather_info(admin_code):
    # 현재 날짜와 시간을 YYYYMMDD 형식으로 가져오기 (시간은 00으로 설정)
    current_datetime = datetime.now().strftime("%Y%m%d%H")  # 오늘 날짜를 가져옵니다.

    # Open API를 통해 자외선 지수 받아오기
    url = f"http://apis.data.go.kr/1360000/LivingWthrIdxServiceV4/getUVIdxV4?serviceKey=D6ujejAVgV7IM%2F5696v5yD82IYnB60%2Bf0I3X2tkWDyp%2BrAUOK39kcfIMPyCI74f1o4Lr7hZhf3HTe9ikDCkaVQ%3D%3D&numOfRows=10&pageNo=1&areaNo={admin_code}&time={current_datetime}&dataType=XML"
    response = requests.get(url)

    if response.status_code == 200:
        uv_root = ET.fromstring(response.text)

        # 필요한 정보 추출
        uv_index_element = uv_root.find(".//h0")
        if uv_index_element is not None:
            uv_index = uv_index_element.text
            print(f"현재 자외선 지수: {uv_index}")

            return float(uv_index) if uv_index.isdigit() else None  # 숫자가 아니라면 None 반환

    return None


@bp.route('/uvindex', methods=['GET', 'POST'])
def get_uv_index():
    if request.method == 'POST':
        user_location = request.form['location'].strip()

        admin_code = get_admin_code(user_location)

        if admin_code:
            decimal_longitude, decimal_latitude = get_decimal_coordinates(admin_code)
        else:
            return render_template('uvindex.html', uv_info=None, map=None)

        # 여기서부터 Folium을 사용하여 지도를 생성합니다.
        map = folium.Map(location=[decimal_latitude, decimal_longitude], zoom_start=12, width='50%',
                         height='50%')  # 입력된 위치로 이동
        # 초기 위치를 서울로 설정

        _strUviWarningLevelInfoMsg = {
            '낮음': "- 햇볕 노출에 대한 보호조치가 필요하지 않음\n - 그러나 햇볕에 민감한 피부를 가진 분은 자외선 차단제를 발라야 함",
            # 위험
            '보통': "- 햇볕에 노출 시 2~3시간 내에 피부 화상을 입을 수 있음\n - 외출 시 모자, 선글라스 이용\n- 자외선 차단제를 발라야 함",
            # 매우 높음
            '높음': "- 햇볕에 노출 시 1~2시간 내에도 피부 화상을 입을 수 있어 위험함\n - 한낮에는 실내나 그늘에 머물러야 함\n- 외출 시 긴 소매 옷, 모자, 선글라스 이용\n- 자외선 차단제를 정기적으로 발라야 함",
            # 높음
            '매우 높음': "- 햇볕에 노출 시 수십 분 이내에도 피부 화상을 입을 수 있어 매우 위험함\n - 오전 10시부터 오후 3시까지 외출을 피하고 실내나 그늘에 머물러야 함\n- 외출 시 긴 소매 옷, 모자, 선글라스 이용\n- 자외선 차단제를 정기적으로 발라야 함",  # 보통
            '위험': "- 햇볕에 노출 시 수십 분 이내에도 피부 화상을 입을 수 있어 가장 위험함\n - 가능한 실내에 머물어야 함\n- 외출 시 긴 소매 옷, 모자, 선글라스 이용\n- 자외선 차단제를 정기적으로 발라야 함"  # 낮음
        }  # 자외선 지수 경고 레벨에 따른 정보 문자열

        uv_info = get_weather_info(admin_code)  # 기상 정보를 가져옵니다.

        if uv_info is not None:
            uv_warning_level = get_uv_warning_level(uv_info)  # 자외선 지수 경고 레벨 계산
            uv_warning_info = _strUviWarningLevelInfoMsg[uv_warning_level]
        else:
            uv_warning_level = '0'  # 자외선 지수가 없는 경우 '0'으로 설정
            uv_warning_info = ''  # 경고 정보가 없는 경우 공백 문자열

        uv_info_dict = {
            'location': user_location,
            'admin_code': admin_code,
            'current_datetime': datetime.now().strftime("%Y%m%d%H"),
            'uv_index': uv_info,
            'uv_warning_level': uv_warning_level,
            'uv_warning_info': uv_warning_info  # 사용자에게 표시될 행동 요령
        }

        # 사용자 입력 위치로 지도 이동
        decimal_longitude, decimal_latitude = get_decimal_coordinates(admin_code)
        map = folium.Map(location=[decimal_latitude, decimal_longitude], zoom_start=12, width='50%',
                         height='50%')  # 입력된 위치로 이동

        # 격자 정보를 기반으로 지도에 마커 추가
        for index, row in excel_data.iterrows():
            if row['행정구역코드'] == admin_code:
                grid_x = row['격자 X']
                grid_y = row['격자 Y']
                uv_index = uv_info

                # 배경색 설정
                level = get_uv_warning_level(uv_index)
                background_color = {
                    '낮음': 'white',
                    '보통': 'yellow',
                    '높음': 'orange',
                    '매우 높음': 'red',
                    '위험': 'black'
                }.get(level, 'blue')  # 기본값은 파란색으로 설정

                # CircleMarker로 마커 추가 (동그라미 원형)
                folium.CircleMarker([decimal_latitude, decimal_longitude], radius=50,
                                    popup=f'격자 위치: {grid_x}, {grid_y}',
                                    color=background_color, fill_color=background_color,
                                    fill_opacity=0.5, fill=True).add_to(map).add_child(
                    folium.Tooltip(f'현재 자외선 지수 : {uv_index}',
                                   style=f'background:{background_color};padding: 30px;; font-size: 17px;'))

            # HTML로 변환
        map_html = map.get_root().render()  # HTML로 변환

        return render_template('uvindex.html', uv_info=uv_info_dict, map=map._repr_html_())
    else:
        return render_template('uvindex.html', uv_info=None, map=None)



if __name__ == '__main__':
    app.run(debug=True)