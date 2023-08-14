import pandas as pd


def google_drive_mount():
    from google.colab import drive
    drive.mount('/content/drive/')

def write_dataframe_to_google_sheet(dataframe, sheet_title, worksheet_title):
    import gspread
    from google.colab import auth
    # Authenticate using your Google account
    auth.authenticate_user()
    
    # Open a connection to Google Sheets
    gc = gspread.authorize(None)  # None parameter uses the already authenticated user
    
    # Open the existing Google Sheet by its title
    existing_sheet = gc.open(sheet_title)
    
    # Create a new worksheet within the existing Google Sheet
    new_worksheet = existing_sheet.add_worksheet(title=worksheet_title, rows=str(dataframe.shape[0]), cols=str(dataframe.shape[1]))
    
    # Update the new worksheet with the DataFrame data
    cell_range = f'A1:{chr(ord("A") + dataframe.shape[1] - 1)}{dataframe.shape[0]}'
    new_worksheet.update(cell_range, dataframe.values.tolist())