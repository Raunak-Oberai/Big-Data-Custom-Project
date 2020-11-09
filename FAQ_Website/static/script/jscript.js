function SearchLoader()
{
    if (validate_empty('question'))
    {
        var twitter_form_data = new FormData();
        var twitter_search_value = document.getElementById('question').value;
        twitter_form_data.append("question",twitter_search_value);
        ajaxCall('search',twitter_form_data,'load');
        parse("json");
    }
}
function parse(spec) {

  vg.parse.spec(spec, function(chart) { chart({el:"#vis"}).update(); });

}

