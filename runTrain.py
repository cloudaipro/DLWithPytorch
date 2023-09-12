import datetime

from util.util import importstr
from util.logconf import logging
log = logging.getLogger('nb')

def run(app, *argv):
    argv = list(argv)
    argv.insert(0, '--num-workers=1')  # <1>
    log.info("Running: {}({!r}).main()".format(app, argv))
    
    app_cls = importstr(*app.rsplit('.', 1))  # <2>
    app_cls(argv).main()
    
    log.info("Finished: {}.{!r}).main()".format(app, argv))

# freeze_support()
run('training.LunaTrainingApp', '--epochs=1')
