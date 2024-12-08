loglevel = "debug"
errorlog = "-"
accesslog = "-"
worker_tmp_dir = "/dev/shm"
graceful_timeout = 120
timeout = 60
keepalive = 5
worker_class = "gthread"
workers = 2
threads = 8
bind = "0.0.0.0:8000"

capture_output = True
enable_stdio_inheritance = True

reload = True

access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
