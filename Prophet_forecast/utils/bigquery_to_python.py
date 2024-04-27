"""
Functions to do the import and export of data between BigQuery and Python.
"""
import time

from google.cloud import bigquery

######################
# bigquery_to_python #
######################
def bigquery_to_python(sia_project, query):
    """Import BigQuery table in Python dataframe"""
    # Create BigQuery client
    client = bigquery.Client(project=sia_project)

    # Fetch data from table and convert to Pandas dataframe
    return client.query(query).to_dataframe()


######################
# python_to_bigquery #
######################
def python_to_bigquery(
    dataframe,
    sia_project,
    dataset_table,
    table_schema,
    write_disposition,
    partition_field=None,
):
    """Export Python dataframe in BigQuery table"""
    # Create BigQuery client
    client = bigquery.Client(project=sia_project)

    schema = []

    for key, value in table_schema.items():
        schema.append(bigquery.SchemaField(key, value))

    if write_disposition == "APPEND":
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            time_partitioning=_get_time_partitioning(partition_field),
        )
    else:
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            time_partitioning=_get_time_partitioning(partition_field),
        )
    job = client.load_table_from_dataframe(
        dataframe, dataset_table, job_config=job_config
    )
    job.result()


def python_to_bigquery_with_retries(
    dataframe,
    sia_project,
    dataset_table,
    table_schema,
    write_disposition,
    partition_field=None,
    nb_attempts=2,
):
    """Export dataframe in BQ table and wait-retry if exceptions occurs"""
    for _ in range(nb_attempts + 1):
        try:
            python_to_bigquery(
                dataframe,
                sia_project,
                dataset_table,
                table_schema,
                write_disposition,
                partition_field,
            )
            break
        except:
            time.sleep(3)
            python_to_bigquery(
                dataframe,
                sia_project,
                dataset_table,
                table_schema,
                write_disposition,
                partition_field,
            )


def _get_time_partitioning(partition_field):
    """Create time partitioning object"""
    if partition_field:
        return bigquery.TimePartitioning(field=partition_field)
    return None