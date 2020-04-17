from kdcovid.search_tool import SearchTool
from kdcovid.task_questions import TaskQuestions

import pickle

from absl import flags
from absl import app
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '2020-04-10', 'data path')
flags.DEFINE_string('out_dir', '2020-04-10', 'out path')
flags.DEFINE_string('css_file', 'style.css', 'css_file')
flags.DEFINE_string('paper_id', 'cord_uid', 'cord_uid or sha')
flags.DEFINE_string('results', None, 'results pickle')

logging.set_verbosity(logging.INFO)


css = """html
{
 overflow: -moz-scrollbars-vertical; 
 overflow-y: scroll;
}

body
{
 margin:0 auto;
 text-align:left;
 width:100%;
 padding: 0 0;
 font-family: 'Basic Sans', sans-serif;
 background-color:#fff;
 color: #3E4550;
}
#wrapper
{
 margin:0 auto;
 text-align:left;
}
#wrapper h1
{
 margin-top:50px;
 font-size:45px;
 font-family: 'Basic Sans', sans-serif;
 color:#585858;
}
#wrapper h1 p
{
 font-size:18px;
 font-family: 'Basic Sans', sans-serif;
}

.q_container {
    display: flex;
}

.q_container {
    display: flex;
    max-width: 900px;
    margin: auto;
    flex-wrap: wrap;
}

.q_inner {
    flex: 1;
}

.q_inner {
    background-color: #fff;
    padding: 10px 20px;
    box-shadow: none;
    border: 2px solid #178EF4;
    margin: 10px;
    color: rgb(0, 61, 114);
    border-radius: 7px;
    cursor: pointer;
    transition: all 0.3s ease;
    align-items: center;
    max-width: 300px;
    display: flex;
    flex: 0 0 calc(33% - 70px);
    text-align: center;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-family: 'Basic Sans', sans-serif;
    /*Added later*/
    -webkit-appearance: none;
    -webkit-rtl-ordering: logical;
    -webkit-writing-mode: unset;
    box-sizing: unset;
    user-select: unset;
    white-space: unset;
    font: unset;
    text-rendering: unset;
    letter-spacing: unset;
    word-spacing: unset;
    text-transform: unset;
    text-indent: unset;
    text-shadow: unset;
}

.q_inner:hover {
    background-color: #178EF4;
    margin: 10px;
    color: #fff;
    cursor: pointer;
}


#search_box input[type="text"]
{
 width:450px;
 height:45px;
 padding-left:10px;
 font-size:18px;
 font-family: 'Basic Sans', sans-serif;
 margin-bottom:15px;
 color:#424242;
 border-radius: 7px;
 border: 2px solid #A9B5C7;

}

.search {
    display: flex;
    justify-content: center;
}
#search_box input[type="submit"]
{
 width:100px;
 height:45px;
 background-color:#178EF4;
 color:white;
 border-radius: 7px;
 border:none;
 margin-left: 20px;
 font-size: 1rem;
 text-transform: none;
 cursor: pointer;
 font-family: 'Basic Sans', sans-serif;
 transition: all 0.3s ease;
 outline: none;
}
#search_box input[type="submit"]:hover
{
 transform: scale(1.1);
 background-color: rgb(16, 123, 216);
 
}

#result_div
{
 margin: auto;
 text-align:left;
}
#result_div li
{
 margin-bottom:20px;
 list-style-type:none;
}
#result_div li a
{
 text-decoration:none;
 display:block;
 text-align:left;
}
#result_div li a .title
{
 font-weight:bold;
 font-size:18px;
 font-family: 'Basic Sans', sans-serif;
 color:#5882FA;
}
#result_div li a .desc
{
 color:#6E6E6E;
}
.topnav {
  background-color: #fff;
  width: 100%;
  overflow: hidden;
}

.container-inner {
    width: 100%;
    max-width: 1200px;
    margin: auto;
    display: flex;
    align-items: center;
}

h1 {
    font-size: 1.5rem;
    color: rgb(97, 114, 141);
    margin: 10px;
    margin-bottom: 30px;
    font-weight:400;
}

.logo img {
    width: 150px;
}

.container-inner div:nth-child(1) {
    flex: 1;
    text-align: left;
}

.res_text h2 ~ i {
    color: #0070D0;
    font-size: 14px;
}

.res_text h2 a {
    color: #3E4550;
    font-weight: 700;
    text-decoration: none;
}

.year {
    color: #0070D0;
    font-size: 14px;
}

.res_text h2 {
    margin: 0.5rem 0px ;
}

.res_image h2 {
    margin: 5px 0px;
    margin-top: 30px;
}
.res_image p {
    margin: 0px;
    font-size: 14px;

}
.jumbotron {
    padding: 100px 20px;
    height: auto;
    padding-bottom: 50px;
}
/* Style the links inside the navigation bar */
.topnav a {
  float: left;
  color: #3E4550;
  text-align: center;
  padding: 14px 16px;
  text-decoration: none;
  font-size: 17px;
}
/* Change the color of links on hover */
.topnav a:hover {
  background-color: #c0e2ff;
  color: black;
}
/* Add a color to the active/current link */
.topnav a.active {
  background-color: #8dd3c7;
  color: white;
}
.round {
  stroke-linejoin: round;
  font-family: 'Basic Sans', sans-serif;
}
.wrap {
  display: flex;
   width: auto;
    max-width: 1200px;
    margin: auto;
    padding: 40px 20px;

}

.res_text h2 i {
    font-size: 16px;
    margin-left: 10px;
}

.res_text h2 a {
    transition: all 0.3s ease;
}
.res_text h2:hover a {
    color: rgb(103, 114, 131);
}

.legend {
    border-top: 1px solid #BBC3CE;
    display: flex;
    margin-top: 20px;
    padding: 20px;
}

.legend div {
    display: flex;
    align-items: center;
}

.entity {
    padding: 0px 3px !important;
    border-radius: 5rem !important;
}

.legend p {
    margin: 0 30px 0px 10px;
}
.circle {
    width: 15px;
    height: 15px;
    border-radius: 30px;
}

.yellow {
    background-color: #FDF8B9;
}
.orange {
    background-color: #FFA07A;
}
.purple {
    background: linear-gradient(90deg, #aa9cfc, #fc9ce7);
}

.res_text {
  flex: 1;
  display: flex;
  flex-direction: column;
   margin: 0px auto;
   line-height: 1.6rem;
    border: 1px solid #BBC3CE;
    border-top-left-radius: 10px;
    border-bottom-left-radius: 10px;
  /* border: 1px solid green; */
}

.paper-details {
    padding: 20px;
    flex: 1;
}



.j-logo {
    max-width: 700px;
    width: 100%;
}

.res_image {
  padding: 20px;
  border-top: 1px solid #BBC3CE;
  border-bottom: 1px solid #BBC3CE;
  border-right: 1px solid #BBC3CE;
  border-top-right-radius: 10px;
  border-bottom-right-radius: 10px;
}

.res_image::-webkit-scrollbar {
    -webkit-appearance: none;
}

.res_image::-webkit-scrollbar:vertical {
    width: 8px;
}

.res_image::-webkit-scrollbar:horizontal {
    height: 8px;
}

.res_image::-webkit-scrollbar-thumb {
    border-radius: 1px;
    border: 1px solid white; /* should match background, can't be transparent */
    background-color: rgba(0, 0, 0, .5);
}

@media (max-width: 767px) {
  .wrap {
    flex-direction: column;
  }
  .one,
  .two {
    width: auto;
    overflow: -moz-scrollbars-vertical; 
    overflow: scroll;
  }

  .res_image object {
    width: 100%;
  }

  .res_image {
    border-bottom-left-radius: 10px;
    border-bottom-right-radius: 10px;
    border-left: 1px solid #BBC3CE;
    border-top: none;
    border-top-right-radius: 0px;
  }

  .res_text {
      border-top-right-radius: 10px;
      border-bottom-right-radius: 0px;
      border-bottom-left-radius: 0px;
  }

  .q_inner {
      flex: 0 0 calc(50% - 70px);
  }
}
@media (max-width: 464px) {
  .q_container {
      flex-direction: column;
  }

  .q_inner {
      flex: 1;
      max-width: none;
  }
}
"""



def format_task_html(task2subtasks, subtasks_html):
    s = ""
    s += "<BR/><BR/><h1>Queries for Provided Questions per Task:</h1><BR/>"
    for t, subtasks in task2subtasks.items():
            s += """<br><br><button type="button" class="collapsible"><b>{}</b></button><div class="content">{}</div>""".format(
                t, format_subtask_html(subtasks, subtasks_html))
    return s


def format_subtask_html(subtasks, subtasks_html):
    r = ''
    for st, _, _ in subtasks:
        r += """<br><br><button type="button" class="collapsible">{}</button>
            <div class="content1">
              <p>{}</p>
            </div>""".format(st, subtasks_html[st])
    return r

def format_example_queries(example_queries, queries_html, css):
    html_string = """
    <!DOCTYPE html>
    <html>
    <head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <link rel="stylesheet" type="text/css" href="./style.css">
    <link rel="icon" type="image/ico" href="./favicon.ico">
    <link rel="stylesheet" href="https://use.typekit.net/wtq0evn.css">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    INSERT_CSS

    .collapsible {
      background-color: white;
      color: #3E4550;
      border-radius: 7px;
      cursor: pointer;
      padding: 18px;
      text-align: left;
      width: 100%
      outline: none;
      font-size: 18px;
      border-style: solid;
      border-color: #3E4550;
      font-family: 'Basic Sans', sans-serif;
    }

    .active, .collapsible:hover {
      background-color: rgb(36.1472, 145.5616, 241.9456);
      color: white;
    }


    .content1 {
      padding: 0 18px;
      display: none;
      overflow: hidden;
      background-color: white;
      font-size: 15px;
      font-family: 'Basic Sans', sans-serif;
      text-align: left;
    }

    .content {
      padding: 0 18px;
      display: none;
      overflow: hidden;
      background-color: white;
      font-size: 15px;
      font-family: 'Basic Sans', sans-serif;
      text-align: left;
    }

    </style>
    </head>
    <body>


    CONTENT

    <script>
    var coll = document.getElementsByClassName("collapsible");
    var i;

    for (i = 0; i < coll.length; i++) {
      coll[i].addEventListener("click", function() {
        this.classList.toggle("active");
        var content = this.nextElementSibling;
        if (content.style.display === "block") {
          content.style.display = "none";
        } else {
          content.style.display = "block";
        }
      });
    }
    </script>

    </body>
    </html>
    """.replace('CONTENT', format_subtask_html(example_queries, queries_html)).replace('INSERT_CSS', css)
    return html_string

def format_tasks(task2subtasks, subtasks_html, css):

    html_string = """
    <!DOCTYPE html>
    <html>
    <head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <link rel="stylesheet" type="text/css" href="./style.css">
    <link rel="icon" type="image/ico" href="./favicon.ico">
    <link rel="stylesheet" href="https://use.typekit.net/wtq0evn.css">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    INSERT_CSS
   
    .collapsible {
      background-color: white;
      color: #3E4550;
      border-radius: 7px;
      cursor: pointer;
      padding: 18px;
      text-align: left;
      width: 100%
      outline: none;
      font-size: 18px;
      border-style: solid;
      border-color: #3E4550;
      font-family: 'Basic Sans', sans-serif;
    }
    
    .collapsible1 {
      background-color: white;
      color: #3E4550;
      border-radius: 7px;
      cursor: pointer;
      padding: 18px;
      text-align: left;
      width: 100%
      outline: none;
      font-size: 18px;
      border-style: solid;
      border-color: #3E4550;
      font-family: 'Basic Sans', sans-serif;
    }


    .active, .collapsible:hover {
      background-color: rgb(36.1472, 145.5616, 241.9456);
      color: white;
    }


    .content1 {
      padding: 0 18px;
      display: none;
      overflow: hidden;
      background-color: white;
      font-size: 15px;
      font-family: 'Basic Sans', sans-serif;
      text-align: left;
    }

    .content {
      padding: 0 18px;
      display: none;
      overflow: hidden;
      background-color: white;
      font-size: 15px;
      font-family: 'Basic Sans', sans-serif;
      text-align: left`;
    }

    </style>
    </head>
    <body>
    
    
    CONTENT
    
    <script>
    var coll = document.getElementsByClassName("collapsible");
    var i;
    
    for (i = 0; i < coll.length; i++) {
      coll[i].addEventListener("click", function() {
        this.classList.toggle("active");
        var content = this.nextElementSibling;
        if (content.style.display === "block") {
          content.style.display = "none";
        } else {
          content.style.display = "block";
        }
      });
    }
    </script>
    
    </body>
    </html>
    """.replace('CONTENT', format_task_html(task2subtasks, subtasks_html)).replace('INSERT_CSS', css)
    return html_string


def main(argv):
    logging.info('Running create_cached_search_results with arguments: %s', str(argv))

    task_questions = TaskQuestions()

    if FLAGS.results is None:
        search_tool = SearchTool(data_dir=FLAGS.data_dir, paper_id_field=FLAGS.paper_id, use_object=False,
                                 gv_prefix='http://kdcovid.nl/', legacy_metadata=True)
        from collections import defaultdict
        results = defaultdict(str)
        for task, questions in task_questions.task2questions.items():
            for q, recent, covid in questions:
                html_res = search_tool.get_search_results(q, recent, covid)
                results[q] = html_res

        for q, recent, covid in task_questions.example_queries:
            html_res = search_tool.get_search_results(q, recent, covid)
            results[q] = html_res

        with open(FLAGS.out_dir + '/cached_results.pkl', 'wb') as fout:
            pickle.dump(results, fout)
    else:
        with open(FLAGS.results, 'rb') as fin:
            results = pickle.load(fin)

    html_output = format_tasks(task_questions.task2questions, results, css)

    with open(FLAGS.out_dir + "/tasks.html", 'w') as fout:
        fout.write(html_output)

    html_examples = format_example_queries(task_questions.example_queries, results, css)
    with open(FLAGS.out_dir + "/examples.html", 'w') as fout:
        fout.write(html_examples)


if __name__ == '__main__':
    app.run(main)
