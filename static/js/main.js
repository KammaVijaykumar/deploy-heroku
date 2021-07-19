$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Predict
    $('#btn-predict').click(function () {
        var username= $('#userName').val();
        console.log(username)
        var jsonData = {loginUserName : username};

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: jsonData,
            dataType: "JSON",
            cache: false,
            async: true,
            success: function (data) {
                // Get and display the result
                console.log( data )
                var len = data.length;
                for(var i=0; i<len; i++){
                    var tr_str = "<tr>" +
                    "<td align='center'>" + data[i] + "</td>" +
                    "</tr>";
                }
                console.log('Success!');
            },
        });
    });

});
