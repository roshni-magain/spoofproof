import streamlit as st
from multiapp import MultiApp
from apps import Training, detector # import your app modules here

app = MultiApp()

from PIL import Image
# DB Management
import sqlite3
conn = sqlite3.connect('../data.db')
c = conn.cursor()

# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username, password))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data


header = st.container()
col2, col1 = st.columns(2)
dashboard = st.container()
with header:
    st.title("Welcome to spoofproof!")
    st.markdown("App for Fake News Detection")
    img = Image.open("spoofproof.png")
    st.image(img)

with col1:
    st.title("About spoofproof")
    st.markdown("**Spoofproof** is a web app that allows "
                "the user to detect fake news on a "
                "dashboard. Spoofproof uses trained **machine "
                "learning algorithms** that detects validity "
                "of news with tailored input.")
with col2:
    st.title("Log in")

    login_choice = ["Log In", "Sign Up"]
    choice = st.selectbox("Enter Dashboard: ", login_choice)

    if choice == 'Log In':
        st.subheader("Log In Here")
        username = st.text_input("User Name")
        password = st.text_input("Password", type='password')
        if st.checkbox("Log In"):
            # if password == '12345':
            create_usertable()
            result = login_user(username, password)
            if result:
                st.success("Logged In as {}".format(username))
                # Add all your application here
                with dashboard:
                    app.add_app("Detector", detector.app)
                    app.add_app("Custom Detector", Training.app)
                    # The main app
                    app.run()
            else:
                st.warning("Incorrect Username/Password")
    elif choice == "Sign Up":
        st.subheader("Create New Account")
        new_user = st.text_input("User Name")
        new_password = st.text_input("Password", type='password')
        if st.checkbox("Sign Up"):
            create_usertable()
            add_userdata(new_user, new_password)
            st.success("You have successfully created an account!")
            st.info("Click the Log In button to continue...")

