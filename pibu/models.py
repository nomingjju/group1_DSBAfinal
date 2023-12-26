from pibu import db

class SocialLogin(db.Model): # 마이페이지
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(11), nullable=False) # 이름이 아이디
    password = db.Column(db.String(11), nullable=False) # 전화번호가 비밀번호
    email = db.Column(db.String(30), nullable=False)
    gender = db.Column(db.Integer, nullable=False) # 남자 : 1, 여자 : 0
    age = db.Column(db.Integer, nullable=False)
    agree_personal = db.Column(db.Integer, nullable=True)
    agree_location = db.Column(db.Integer, nullable=True)
    skinImage = db.Column(db.String(50), nullable=True)

class SkinCancerResult(db.Model): # 피부암 진단
    id = db.Column(db.Integer, primary_key=True)
    cancerList = db.Column(db.String(20), nullable=False)
    cancerDesc = db.Column(db.String(100), nullable=False)
    pibuMap = db.Column(db.String(100), nullable=False)

class SkinCancer(db.Model): # 피부암 데이터
    id = db.Column(db.Integer, primary_key=True) # 오토인크리먼트로 숫자 자동 생성

    social_login_id = db.Column(db.Integer, db.ForeignKey('social_login.id', ondelete='CASCADE'))
    social_login = db.relationship('SocialLogin', backref=db.backref('skin_cancer_set'))

    skin_cancer_result_id = db.Column(db.Integer, db.ForeignKey('skin_cancer_result.id', ondelete='CASCADE'))
    skin_cancer_result = db.relationship('SkinCancerResult', backref=db.backref('skin_cancer_set'))