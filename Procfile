web: python cot_app.py
web: gunicorn app:app --workers 4 --worker-class gevent --timeout 60 --preload --worker-connections 1000

