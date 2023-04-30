window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        update_search: function (value) {
            if (window.location.pathname == "/topic_details/" && value != "NO UPDATE") {
                if (history.pushState) {
                    var newurl = window.location.protocol + "//" + window.location.host + window.location.pathname + value;
                    window.history.pushState({ path: newurl }, '', newurl);
                }
            }
            return '';
        }
    }
});