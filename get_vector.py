wweight, words_index = pickle.load(open('wd.pkl','rb'))
word_vect = {i:wweight[words_index[i],:] for i in words_index}
pickle.dump(word_vect, open('word_data.pkl','wb'))
