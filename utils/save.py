import datetime


def get_savefile_name(epoch):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"tinyTransformer_{current_date}_epoch{epoch}.pt"
