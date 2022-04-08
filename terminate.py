import signal
import sys

stop_training = False


def signal_handler(sig, frame):
    print('\nDetected Ctrl+C, stopping training')
    stop_training = True
    print('Saving model')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

print('Starting training')
while(True):
    # if stop_training:
    #     break
    pass
