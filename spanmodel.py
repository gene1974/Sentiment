import ltp

if __name__ == '__main__':
    texts = ['这些功能将长期维护，通常不应有重大的性能限制或文档中的空白。我们还希望保持向后兼容性（尽管可能会发生重大更改，并且会提前发布一个版本）']
    ltp_tool = ltp.LTP()
    # output: word_cls, word_input, word_length, word_cls_input, word_cls_mask
    segs, hidden = ltp_tool.seg(texts)
    ner = ltp_tool.ner(hidden)
    print(segs)
    print(ner)


# # user_dict.txt 是词典文件， max_window是最大前向分词窗口
# ltp.init_dict(path="user_dict.txt", max_window=4)
# # 也可以在代码中添加自定义的词语
# ltp.add_words(words=["负重前行", "长江大桥"], max_window=4)
