# -*- coding: utf-8 -*-

import re
import sys
from compatitator import strdecode


def extract(sentence, list_pattern, list_pre_pattern=None):
    for k, pattern in enumerate(list_pattern):
        if list_pre_pattern:
            ptn = list_pre_pattern[k]
        else:
            ptn = re.compile(u"^" + pattern.replace(u"[TARGET]", u"(.+?)") + u"$")
        mch = ptn.match(sentence)
        if not mch:
            continue

        for grp in mch.groups():
            print("%s\t%s\t%s" % (grp, sentence, pattern)).encode("utf-8")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit(1)

    list_ptn = list()
    with open(sys.argv[1], "r") as fin:
        for line in fin:
            list_ptn.append(strdecode(line.strip().split("\t")[0]))
    fin.close()

    list_pre_ptn = list()
    for item in list_ptn:
        ptn = re.compile(u"^" + item.replace(u"[TARGET]", u"(.+?)") + u"$")
        list_pre_ptn.append(ptn)

    for line in sys.stdin:
        extract(strdecode(line.strip()), list_ptn, list_pre_ptn)
