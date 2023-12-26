from flask import Blueprint, url_for, render_template
from pibu.models import SocialLogin
from werkzeug.utils import redirect

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return redirect(url_for('uvmuhwan._list'))