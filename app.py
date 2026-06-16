import streamlit as st
import pandas as pd

# Mengambil dataset Twitch langsung dari GitHub kelompok Anda
url = "https://raw.githubusercontent.com/DachsteinSilalahi/Dataset_kelompok2/refs/heads/main/twitchdata-update.csv"
original_df = pd.read_csv(url)

st.set_page_config(layout="wide")
st.title('Twitch Streamer Info Viewer')
st.write('Aplikasi ini menampilkan informasi streamer dari dataset Twitch.')

st.header('Informasi Streamer dari Dataset')
st.write('Pilih streamer dari dataset untuk melihat status Partnered, Mature, dan metrik lainnya.')

selected_channel = st.selectbox(
    'Pilih Channel Streamer',
    options=['Pilih Streamer'] + original_df['Channel'].tolist()
)

if selected_channel != 'Pilih Streamer':
    streamer_data = original_df[original_df['Channel'] == selected_channel].iloc[0]
    st.subheader(f'Detail untuk Channel: {selected_channel}')
    
    col1, col2 = st.columns(2)
    with col1:
        actual_partnered_status = streamer_data['Partnered']
        if actual_partnered_status:
            st.success('Status Partnered: Partner (True)')
        else:
            st.warning('Status Partnered: Bukan Partner (False)')
    with col2:
        is_mature = streamer_data['Mature']
        if is_mature:
            st.error('Status Mature: Konten Dewasa (True)')
        else:
            st.info('Status Mature: Konten Umum (False)')
            
    st.write('---')
    st.subheader('Fitur/Metrik Streamer:')
    st.write(streamer_data.drop(['Channel', 'Partnered', 'Mature']))