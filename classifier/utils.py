def logwrite(log, s, to_print=True):
    if to_print:
        print(s)
    log.write(str(s) + "\n")