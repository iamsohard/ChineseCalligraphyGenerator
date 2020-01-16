from django.contrib import admin
from .models import History

# Register your models here.
class HistoryAdmin(admin.ModelAdmin):
    list_display = ('history_id', 'content', 'calligraphy', 'timeStamp', 'info')

admin.site.register(History, HistoryAdmin)