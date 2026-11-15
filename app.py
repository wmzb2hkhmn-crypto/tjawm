import streamlit as st
import librosa
import numpy as np
import pretty_midi
import tempfile
import os


st.title("🎵 音声 to MIDI 変換ツール")
st.write("スマホやPCから音声ファイル（WAV/MP3等）をアップロードして、MIDIファイルに変換できます。")


def cleanup_file(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def add_note(instrument, pitch, start_time, end_time):
    if pitch is None or end_time <= start_time:
        return
    if end_time - start_time <= 0.05:
        return
    note = pretty_midi.Note(
        velocity=100,
        pitch=int(pitch),
        start=start_time,
        end=end_time,
    )
    instrument.notes.append(note)


if "audio_temp_path" not in st.session_state:
    st.session_state.audio_temp_path = None

uploaded_file = st.file_uploader("音声ファイルを選択してください", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1]
    tmp_path = None

    if st.session_state.audio_temp_path and st.session_state.audio_temp_path != uploaded_file.name:
        cleanup_file(st.session_state.audio_temp_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.session_state.audio_temp_path = tmp_path

    if st.button("変換を実行する", type="primary"):
        output_path = None
        with st.spinner("音声を解析中...しばらくお待ちください"):
            try:
                y, sr = librosa.load(tmp_path, sr=None, mono=True)
                if y.size == 0:
                    raise ValueError("音声データが空です。別のファイルをお試しください。")

                f0, _, _ = librosa.pyin(
                    y,
                    fmin=librosa.note_to_hz("C2"),
                    fmax=librosa.note_to_hz("C7"),
                    sr=sr,
                    hop_length=512,
                )

                midi = pretty_midi.PrettyMIDI()
                instrument = pretty_midi.Instrument(program=0)

                time_step = 512 / sr
                current_pitch = None
                start_time = 0.0

                if np.all(np.isnan(f0)):
                    raise ValueError("音声からピッチを検出できませんでした。別の音声ファイルをお試しください。")

                for i, pitch in enumerate(f0):
                    t = i * time_step

                    if np.isnan(pitch):
                        if current_pitch is not None:
                            end_time = t
                            add_note(instrument, current_pitch, start_time, end_time)
                            current_pitch = None
                        continue

                    midi_num = int(np.round(librosa.hz_to_midi(pitch)))
                    if current_pitch is None:
                        current_pitch = midi_num
                        start_time = t
                    elif current_pitch != midi_num:
                        end_time = t
                        add_note(instrument, current_pitch, start_time, end_time)
                        current_pitch = midi_num
                        start_time = t

                if current_pitch is not None:
                    end_time = len(f0) * time_step
                    add_note(instrument, current_pitch, start_time, end_time)

                if len(instrument.notes) == 0:
                    raise ValueError("有効な音符を検出できませんでした。音声が無音または短すぎる可能性があります。")

                midi.instruments.append(instrument)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as output_file:
                    output_path = output_file.name
                midi.write(output_path)

                st.success("変換が完了しました！")

                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 MIDIファイルをダウンロード",
                        data=f,
                        file_name="converted_output.mid",
                        mime="audio/midi",
                    )
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
            finally:
                cleanup_file(tmp_path)
                cleanup_file(output_path)
                st.session_state.audio_temp_path = None
