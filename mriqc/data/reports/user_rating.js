<script type="text/javascript">
var sentOnce = false;

function setAllRatings(allRatings) {
    allRatings['_items'].sort(function(a, b) {
        if(a['_id'] < b['_id']) return -1;
        if(a['_id'] > b['_id']) return 1;
        return 0;
    });
    console.log(allRatings);
    document.getElementById("allRatings0").textContent = allRatings['_items'][0]['count'];
    document.getElementById("allRatings1").textContent = allRatings['_items'][1]['count'];
    document.getElementById("allRatings2").textContent = allRatings['_items'][2]['count'];
    document.getElementById("allRatings3").textContent = allRatings['_items'][3]['count'];
    document.getElementById("allRatingsDiv").classList.remove("hidden");
};

function sendRating() {
    var allRatings;
    var rating;
    var radios = document.getElementsByName("inlineRadio");
    for (var i = 0; i < radios.length; i++) {
        if (radios[i].checked) {
            rating = radios[i].value;
            break;
        }
    }
    var md5sum = {{ provenance.md5sum }};
    var name = document.getElementById("ratingName").value;
    var comment = document.getElementById("ratingComment").value;
    var params = {'rating': rating, 'md5sum': md5sum, 'name': name, 'comment': comment};

    var ratingReq = new XMLHttpRequest();
    var allRatingsReq = new XMLHttpRequest();
    var aggParam = '{\"$value\":\"' + md5sum + '\"}';
    allRatingsReq.open("GET", "http://0.0.0.0/api/v1/rating_counts?aggregate=" + aggParam);
    allRatingsReq.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    allRatingsReq.setRequestHeader("Authorization", "<secret_token>");
    ratingReq.open("POST", "http://0.0.0.0/api/v1/rating");
    ratingReq.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    ratingReq.setRequestHeader("Authorization", "<secret_token>");
    ratingReq.onload = function () {
        status = ratingReq.status;
        if (status === "201") {
            sentOnce = true;
            allRatingsReq.send();
            setAlert();
        }
    };
    allRatingsReq.onload = function () {
        console.log(allRatingsReq.response);
        allRatings = JSON.parse(allRatingsReq.response);
        setAllRatings(allRatings);
    };
    ratingReq.send(JSON.stringify(params));
};

function setAlert() {
    alertElem = document.getElementById("submitResAlert");
    alertElem.textContent = "Submitted"
    alertElem.className = "alert alert-success"

};

</script>
