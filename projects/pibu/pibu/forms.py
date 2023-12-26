from flask_wtf import FlaskForm
from wtforms import IntegerField, StringField, PasswordField, EmailField
from wtforms.validators import DataRequired, Length, EqualTo, Email

class SocialLoginForm(FlaskForm): # 회원가입
    name = StringField('이름', validators=[DataRequired(), Length(min=3, max=25)])
    password = StringField('전화번호', validators=[DataRequired()])
    email = EmailField('이메일', validators=[DataRequired(), Email()])
    gender = IntegerField('성별', validators=[DataRequired()])  # 남자 : 1, 여자 : 0
    age = IntegerField('나이', validators=[DataRequired()])
    agree_personal = IntegerField('동의', validators=[DataRequired()])
    agree_location = IntegerField('동의', validators=[DataRequired()])

class UserLoginForm(FlaskForm): # 로그인
    name = StringField('사용자이름', validators=[DataRequired(), Length(min=3, max=25)])
    password = StringField('전화번호', validators=[DataRequired()])