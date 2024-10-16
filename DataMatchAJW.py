import pandas as pd
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import streamlit as st
import io
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from functools import partial

def load_unique_values(file, min_value_length):
    """Load unique values for each column from CSV or Excel file."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please use CSV or Excel.")
            return {}
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return {}
    
    unique_values = {}
    for column in df.columns:
        col_data = df[column].dropna()
        if col_data.empty:
            continue
        # Convert all values to strings and strip whitespaces
        col_values = col_data.astype(str).str.strip()
        # Filter values by minimum length
        col_values = col_values[col_values.str.len() >= min_value_length]
        if not col_values.empty:
            unique_values[column] = set(col_values.unique())
    return unique_values

def get_column_type(values):
    """Determine column type based on unique values."""
    if all(v.isdigit() for v in values):
        return 'numeric_string'
    elif all(isinstance(v, str) for v in values):
        return 'text'
    else:
        return 'other'

def compare_columns(col1, values1, col2, values2, match_threshold, min_common_uniques):
    """Compare two columns for exact matches."""
    type1 = get_column_type(values1)
    type2 = get_column_type(values2)
    
    if type1 != type2 and not ({type1, type2} <= {'numeric_string', 'text'}):
        return None
    
    if not values1 or not values2:
        return None
    
    common_values = values1.intersection(values2)
    if len(common_values) >= min_common_uniques:
        match_percentage = len(common_values) / min(len(values1), len(values2)) * 100
        if match_percentage >= match_threshold:
            return {
                'Target Column': col1,
                'PortCo Column': col2,
                'Data Type': f"{type1}/{type2}",
                'Match Percentage': match_percentage,
                'Common Unique Values': len(common_values),
                'Common Values Sample': ', '.join(list(common_values)[:5])  # Sample of common values
            }
    return None

def compare_columns_wrapper(args):
    """Wrapper function to unpack arguments for compare_columns."""
    return compare_columns(*args)

@st.cache_data(show_spinner=False)
def find_matching_columns(unique_values1, unique_values2, match_threshold, min_common_uniques):
    """Find matching columns between two sets of unique values."""
    column_pairs = [
        (
            col1,
            unique_values1[col1],
            col2,
            unique_values2[col2],
            match_threshold,
            min_common_uniques
        )
        for col1 in unique_values1
        for col2 in unique_values2
    ]

    results = []
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for result in executor.map(compare_columns_wrapper, column_pairs):
            if result is not None:
                results.append(result)
    return results

def main():
    st.title("Column Matcher App")

    # Sidebar for thresholds
    with st.sidebar:
        st.header("Set Thresholds")
        match_threshold = st.number_input(
            "Match Percentage Threshold (%)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.1,
            format="%.2f"
        )
        min_common_uniques = st.number_input(
            "Minimum Number of Common Unique Values",
            min_value=0,
            max_value=100000,
            value=10,
            step=1
        )
        min_value_length = st.number_input(
            "Minimum Length of String Values",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )
        page_size = st.selectbox(
            "Items per page",
            options=[25, 50],
            index=0,
        )

    st.header("Upload Files")
    col1, col2 = st.columns(2)

    with col1:
        portco_file = st.file_uploader(
            "Upload PortCo File", type=['csv', 'xlsx', 'xls'], key='portco'
        )
    with col2:
        target_file = st.file_uploader(
            "Upload Target File", type=['csv', 'xlsx', 'xls'], key='target'
        )

    if portco_file and target_file:
        # Option to preview data
        preview_data = st.checkbox("Preview Uploaded Data")
        if preview_data:
            with st.expander("PortCo Data Preview"):
                st.dataframe(pd.read_csv(portco_file, low_memory=False) if portco_file.name.endswith('.csv') else pd.read_excel(portco_file))
            with st.expander("Target Data Preview"):
                st.dataframe(pd.read_csv(target_file, low_memory=False) if target_file.name.endswith('.csv') else pd.read_excel(target_file))

        run_analysis = st.button("Run", disabled=preview_data)

        if run_analysis:
            st.info("Processing...")
            unique_values_target = load_unique_values(target_file, min_value_length)
            unique_values_portco = load_unique_values(portco_file, min_value_length)

            if unique_values_target and unique_values_portco:
                matches = find_matching_columns(
                    unique_values_target,
                    unique_values_portco,
                    match_threshold,
                    min_common_uniques
                )

                if matches:
                    results_df = pd.DataFrame(matches)
                    results_df.sort_values(by='Match Percentage', ascending=False, inplace=True)

                    # Format 'Match Percentage' as percentages
                    results_df['Match Percentage'] = results_df['Match Percentage'].map("{:.2f}%".format)

                    # Store results in session state
                    st.session_state['results_df'] = results_df
                else:
                    st.warning("No matching columns found with the given thresholds.")
                    st.session_state['results_df'] = None
            else:
                st.error("Failed to process the uploaded files. Please check the file formats and content.")
                st.session_state['results_df'] = None

        # Check if results are stored in session state
        if 'results_df' in st.session_state and st.session_state['results_df'] is not None:
            results_df = st.session_state['results_df']

            # Display using AgGrid for interactive filtering and sorting
            gb = GridOptionsBuilder.from_dataframe(results_df)
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=page_size)
            gb.configure_default_column(
                groupable=True,
                value=True,
                enableRowGroup=True,
                editable=False,
                filter=True,
                sortable=True,
            )
            # Configure 'Match Percentage' column to align right and format as percentage
            gb.configure_column("Match Percentage", type=["rightAligned"])
            gridOptions = gb.build()

            st.subheader("Matching Columns")
            AgGrid(
                results_df,
                gridOptions=gridOptions,
                enable_enterprise_modules=False,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                fit_columns_on_grid_load=True,
                theme='streamlit',  # Use 'streamlit' theme
                height=800,  # Increased grid height
            )

            # Provide download button
            def to_excel(df):
                output = io.BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                writer.book.use_zip64()
                writer.close()
                return output.getvalue()

            df_xlsx = to_excel(results_df)

            st.download_button(
                label="Download data as Excel",
                data=df_xlsx,
                file_name='column_matches.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )
        elif run_analysis and st.session_state.get('results_df') is None:
            st.warning("No matching columns found with the given thresholds.")
    else:
        st.info("Please upload both PortCo and Target files to proceed.")

if __name__ == "__main__":
    main()
