# Tensorflow 2.0.0 works with Python 3.7. Python 3.8 generates a `Could not find a version that satisfies the
# requirement tensorflow==2.0.0` error message.

# base image
FROM python:3.7

RUN pip install tensorflow_gpu==2.1
RUN pip install ktrain

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

# exposing default port for streamlit
EXPOSE 8501

# copy over and install packages
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# copying everything over
COPY . .

# run app
CMD streamlit run app.py