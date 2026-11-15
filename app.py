import streamlit as st
import librosa
import numpy as np
import pretty_midi
import tempfile
import os

st.title("🎵 音声 to MIDI 変換ツール")
st.write("スマホやPCから音声ファイル（WAV/MP3等）をアップロードして、MIDIファイルに変換できます。")

# ファイルアップロード
uploaded_file = st.file_uploader("音声ファイルを選択してください", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    # 一時ファイルとして保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if st.button("変換を実行する", type="primary"):
        with st.spinner("音声を解析中...しばらくお待ちください"):
            try:
                # 音声の読み込みとピッチ解析
                y, sr = librosa.load(tmp_path, sr=None)
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr,
                    hop_length=512
                )
                
                # MIDIデータの構築
                midi = pretty_midi.PrettyMIDI()
                instrument = pretty_midi.Instrument(program=0) # ピアノ
                
                time_step = 512 / sr
                current_note = None
                start_time = 0
                
                for i, pitch in enumerate(f0):
                    t = i * time_step
                    if not np.isnan(pitch):
                        midi_num = int(np.round(librosa.hz_to_midi(pitch)))
                        if current_note is None:
                            current_note = midi_num
                            start_time = t
                        elif current_note != midi_num:
                            end_time = t
                            if end_time - start_time > 0.05:
                                note = pretty_midi.Note(velocity=100, pitch=current_note, start=start_time, end=end_time)
                                instrument.notes.append(note)
                            current_note = midi_num
                            start_time = t
                    else:
                        if current_note is not None:
                            end_time = t
                            if end_time - start_time > 0.05:
                                note = pretty_midi.Note(velocity=100, pitch=current_note, start=start_time, end=end_time)
                                instrument.notes.append(note)
                            current_note = None
                            
                if current_note is not None:
                    end_time = len(f0) * time_step
                    note = pretty_midi.Note(velocity=100, pitch=current_note, start=start_time, end=end_time)
                    instrument.notes.append(note)
                    
                midi.instruments.append(instrument)
                output_path = "output.mid"
                midi.write(output_path)
                
                st.success("変換が完了しました！")
                
                # ダウンロードボタン
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 MIDIファイルをダウンロード",
                        data=f,
                        file_name="converted_output.mid",
                        mime="audio/midi"
                    )
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
