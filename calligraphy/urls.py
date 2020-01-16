from django.urls import path

from . import views

app_name = 'calligraphy'

urlpatterns = [
    # 这边是使用通用视图的代码
    path('', views.index, name='index'),
    path('test', views.test, name='test'),
    path('history', views.history, name='history'),
    path('check_txt/', views.check_txt, name='check_txt'),

    # path('detail/', views.detail, name='detail'),
    path('<int:history_id>/', views.detail, name='detail'),
    path('<int:history_id>/delete', views.delete_txt, name='delete_txt'),

    # path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    # path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
    # path('<int:question_id>/vote/', views.vote, name='vote'),
]