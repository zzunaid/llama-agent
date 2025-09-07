import streamlit as st
import os

# Get database URL from secrets
DB_URL = st.secrets.get("DATABASE_URL")
