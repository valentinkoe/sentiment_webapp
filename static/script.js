    function getSentiment() {

        var txt = document.getElementById('textarea').value.trim();

        if (txt.length === 0) {
            alert('No text specified!');
            return;
        }
        
        var trainingData = '';
        
        if (document.getElementById('chkbxImdb').checked && document.getElementById('chkbxYelp').checked) {
            trainingData = 'comb';
        } else if (document.getElementById('chkbxImdb').checked) {
            trainingData = 'imdb';
        } else if (document.getElementById('chkbxYelp').checked) {
            trainingData = 'yelp';
        }
        
        if (trainingData.length === 0) {
            alert('No training data selected!');
            return;
        }
        
        setLoading('overallResult');
        setLoading('bayesText');
        setLoading('svmText');
        setLoading('rulesText');

        jQuery.post('/_get_sentiment', {'txt': txt,
                                        'colorPositive': document.getElementById('colorPositive').value,
                                        'colorNegative': document.getElementById('colorNegative').value,
                                        'colorMixed': document.getElementById('colorMixed').value,
                                        'colorNegation': document.getElementById('colorNegation').value,
                                        'trainingData': trainingData,
                                        'posWords': document.getElementById('posWords').value.trim(),
                                        'negWords': document.getElementById('negWords').value.trim()
                                        }, function (data) {
            
            setText('overallResult',data.result);
            setText('bayesText',data.bayesHTML);
            setText('svmText',data.svmHTML);
            setText('rulesText',data.rulesHTML);

        });

    }

    function setLoading(id) {
        setText(id,'<img src="/static/images/loader.gif" width="25px" height="25px" alt="loading...">');
    }

    function setText(id,text) {
        document.getElementById(id).innerHTML = text;
    }
