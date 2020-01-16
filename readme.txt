1,启动一个完整的django项目的命令是
python manage.py runserver
或

python manage.py runserver 0.0.0.0:8889
python manage.py runserver 9009

2，创建应用
python manage.py startapp polls
应用和mysite有区别

创建完app之后还要再mysite/settings.py里的INSTALLED_APPS配置一下

3，改变模型需要这三步：

编辑 models.py 文件，改变模型。
运行 python manage.py makemigrations 为模型的改变生成迁移文件。
运行 python manage.py migrate 来应用数据库迁移。

4，Django 管理页面

Username: admin
Email address: admin@example.com
Password: 123456
Password (again): 123456
Superuser created successfully.

5，常见错误
"CSRF token missing or incorrect."的解决方法
如果不屏蔽CSRF
html中的form添加模板标签{% csrf_token %}

如果要屏蔽CSRF

方法1：注释掉django工程settings.py中

#'django.middleware.csrf.CsrfViewMiddleware'


方法2：django工程views.py添加屏蔽装饰器

from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def some_view(request):
    #...

change