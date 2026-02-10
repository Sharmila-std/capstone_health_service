# extensions.py
from flask_mail import Mail
from flask_socketio import SocketIO
from apscheduler.schedulers.background import BackgroundScheduler

mail = Mail()
socketio = SocketIO(async_mode='eventlet', cors_allowed_origins="*")
scheduler = BackgroundScheduler()
