let matches = [];

const inputElement = (userInput) => {
    return (
        [
            '<div class="input-element">',
            '<p>',
            userInput,
            '</p>',
            '<button class="delete-button">Delete</button>',
        ].join("")
    )
}

const updateMatches = () => {
}

const addMatch = () => {
    var userInput = $("#matching-input").val();
    if (!userInput.length > 0) {
        return;
    }
    matches.push(userInput);
    $('#matches').append(inputElement(userInput));
    $(this).val('');
}

const deleteMatch = (element) => {
    matches = matches.filter(function (item) {
        return item != $(element).parent('.input-element').text().replace("Delete", "")
    });
    $(element).parent('.input-element').remove();
}

$(document).ready(function () {
    $("#matching-input").on("keyup", function (event) {
        if (event.keyCode === 13) {
            event.preventDefault();
            addMatch();
        }
    });

    $("#matches").on('click', '.delete-button', function () {
        deleteMatch(this);
    });

    $("#add-match").click(function (event) {
        event.preventDefault();
        addMatch();
    });
});