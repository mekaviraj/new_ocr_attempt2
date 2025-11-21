def generate_html(patient, date, df):
    html = f"""
    <html>
    <head>
        <title>OCR Prescription Output</title>
    </head>
    <body>
        <h2>Patient: {patient}</h2>
        <h3>Date: {date}</h3>

        {df.to_html(index=False)}

    </body>
    </html>
    """

    with open("output.html", "w") as f:
        f.write(html)

    print("Generated: output.html")
