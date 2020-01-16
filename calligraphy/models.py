from django.db import models

# Create your models here.
class History(models.Model):
    """
    一次检索记录
    字段
    txt_id id
    news_content 上传文本内容
    news_ans 处理结果
    timeStamp 时间戳
    info: 包含具体一些别的检测信息
    """
    history_id = models.CharField(max_length=200)
    content = models.CharField(max_length=20000)
    calligraphy = models.CharField(max_length=20000)
    timeStamp = models.CharField(max_length=200)
    info = models.CharField(max_length=1000)

    def __str__(self):
        return str(self.content)+str(self.calligraphy)
