from text_process import line_processing
import numpy as np
import random, pickle, traceback

def word_process(text, min_count = 4):
    word_count = {}
    total_wcount = 0
    for line in text.split('\n'):
        lwords = line.split(' ')#.strip('')
        total_wcount += len(lwords)
        for w in lwords:
            if w in word_count:
                word_count[w] += 1
            else:
                word_count[w] = 1
    print len(word_count)
    words = [w for w in word_count]
    low_word = []
    for i in words:
        if word_count[i] < min_count:
            word_count.pop(i)
            low_word.append(i)
    print len(word_count)
    words_index = {}
    count = 0
    for i in word_count: words_index[i] = count;count+=1
    return words_index, low_word, total_wcount

def remove_words(text, words, words_index):
    otext = ''
    count = 0
    lines = text.split('\n')
    wilen = len(words_index)
    a = [i for i in words_index] + words
    twords_index = {a[i]:i for i in range(len(a))}
    index_word ={twords_index[i] : i for i in twords_index}
    print "%d lines to process.."%(len(lines))
    for line in text.split('\n'):
        lwords = line.split(' ')
        lindex = [twords_index[i] for i in lwords]
        ptext = ' '.join([index_word[i] for i in lindex if i < wilen])
        otext +=  ptext+'\n'
        count += 1
        if count % 1000==0: print "%d lines are processed"%(count)
    return otext

def rmsprop(w, dw, m, b=None, db=None, extra=None, b1=.99, neta=.001, e=1e-8):
    # m = self.opt_var.getm(extra)
    m = b1*m + (1 - b1)*np.square(dw)
    w -= neta * np.divide(dw, (np.sqrt(m) + e))
    # self.opt_var.setm(m ,extra)
    # if b is not None:
    #     b -= self.neta * db
    return w, m

def Skip_gram(text, words_index, wcount, twcount, context=3, epoch = 10, d=50, neg=5, neta=.05):
    wweight = np.random.normal(0, .5, (wcount, d))
    sneta = neta
    m = np.zeros((wcount, d))
    # dw = np.zeros((wcount, d))
    count = err = 0
    text = text.strip('\n| ')
    for ep in range(epoch):
        for line in text.split('\n'):
            lwords = [' ' for i in range(context)] + line.split(' ') + [' ' for i in range(context)]
            llen = len(lwords)

            inc=context
            while inc < llen-context:
                w0 = words_index[lwords[inc]]
                wcontext = [words_index[lwords[inc-i-1]] for i in range(context)] + [words_index[lwords[inc+i+1]] for i in range(context)]
                # h = np.sum(wweight[[i for i in wcontext],:],axis=0)/(2*context)
                h = wweight[w0,:]
                for wi in wcontext:
                    w = [wi] + [random.randrange(wcount) for i in range(neg)]
                    wh = [np.exp(np.dot(h,wweight[nw,:].T)) for nw in w]
                    wsum = np.sum(wh)
                    ywh = [i / wsum for i in wh]
                    err -= np.log(ywh[0])
                    twh = [1] + [0 for i in range(neg)]
                    ##################################################################
                    ## Rmsprop
                    for i in range(neg + 1): wweight[w[i], :], m[w[i], :] = rmsprop(w=wweight[w[i], :], dw=((ywh[i] - twh[i]) * h), m=m[w[i], :], neta=neta)
                    ##################################################################
                count += 1
                inc += 1
                if count % 10000 == 0:
                    print "%d error is : %f , %f"%(count, err/100000, neta)
                    # neta = sneta * (1 - count / (float)(epoch * tcword + 1))
                    neta = sneta * 0.0001 if neta < sneta * 0.0001 else sneta * (1 - count / float((epoch * twcount + 1)))
                    err = 0
        print "%d/%d epoch complete"%(ep,epoch)
    pickle.dump([wweight, words_index], open('sk_wd.pkl','wb'))
    return


if __name__ == '__main__':

    text = ''
    count = 0
    for i in open('/media/zero/41FF48D81730BD9B/all-the-news/texted.txt'):
        text+=line_processing(i)+'\n'
        count += 1
        if count % 1000 == 0: break
    words_index, low_word, twcount = word_process(text)
    wcount = len(words_index)
    text = text.strip('\n')
    text = remove_words(text, low_word, words_index)
    text = text.strip('\n')
    words_index[' '] = wcount
    wcount += 1
    print
    Skip_gram(text, words_index=words_index, wcount=wcount, twcount=twcount, neta=.05)


