import re

def kind(t):
    """Rough event type from the title (keyword based)."""
    if re.search(r"演唱會|音樂會|巡迴|Tour|TOUR|LIVE|Live|紅白|音樂節|FanMeeting|見面會|fan", t): return "演唱會/音樂類"
    if re.search(r"賽|聯盟|盃|杯|公開|羽球|籃|排球|網球|冰上|體操|拳|格鬥|SBL|HBL|PLG|T1|UFC", t): return "體育"
    return "其他"
