from flask import Flask, render_template, request, send_file, url_for, redirect, session
from flask_session import Session
from dotenv import load_dotenv
import os
import asyncio
import configuration
import io
import docx

load_dotenv('apiKeys.env')

import agenticReporter

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
app = Flask(__name__, template_folder=template_dir)

app.secret_key = os.urandom(12).hex()

app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        topic = request.form.get("topic")
        try:
            # Build input data and configuration
            input_data = {"topic": topic}
            config = configuration.Configuration(
                report_structure={"introduction": True, "body": True, "conclusion": True},
                number_of_queries=3,
                tavily_topic="general",
                tavily_days=30,
            )
            config_dict = vars(config)

            # Run asynchronous report generation
            result = asyncio.run(agenticReporter.graph.ainvoke(input_data, config_dict))
            final_report = result.get("final_report", "No report generated.")

            # Save report content in the session, so it can be downloaded later
            session['report'] = final_report

            return render_template("result.html", report=final_report)
        except Exception as e:
            return f"An error occurred: {str(e)}"
    return render_template("index.html")


@app.route("/download", methods=["GET"])
def download():
    # Retrieve the generated report from the session
    report_content = session.get('report')
    if not report_content:
        print("No report found in session")
        return redirect(url_for('index'))

    # Create a DOCX document using python-docx
    doc = docx.Document()
    doc.add_paragraph(report_content)

    # Save the document to an in-memory bytes buffer
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="report.docx",
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


if __name__ == "__main__":
    app.run(debug=True)
