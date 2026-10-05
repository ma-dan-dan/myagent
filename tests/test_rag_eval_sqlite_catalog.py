import sqlite3

from app.rag_eval.sqlite_catalog import SQLiteCatalog


def test_sqlite_catalog_uses_database_namespace_and_filters_sensitive_samples(tmp_path):
    database = tmp_path / "shop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders (order_id INTEGER, status TEXT, api_key TEXT)")
        connection.execute("INSERT INTO orders VALUES (1, 'paid', 'sk-secret')")

    records = SQLiteCatalog(sample_limit=3).load("shop", database)

    assert len(records) == 1
    assert records[0].namespace == "shop"
    assert records[0].table_id == "shop.orders"
    assert records[0].columns[1].sample_values == ["paid"]
    assert records[0].columns[2].sample_values == []


def test_sqlite_catalog_keeps_same_named_tables_in_separate_namespaces(tmp_path):
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    for path, value in ((first, "first"), (second, "second")):
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE orders (status TEXT)")
            connection.execute("INSERT INTO orders VALUES (?)", (value,))

    catalog = SQLiteCatalog()
    first_record = catalog.load("first", first)[0]
    second_record = catalog.load("second", second)[0]

    assert (first_record.namespace, first_record.table_id) == ("first", "first.orders")
    assert (second_record.namespace, second_record.table_id) == ("second", "second.orders")
