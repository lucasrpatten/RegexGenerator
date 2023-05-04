let matches = [];
let rejections = [];

const inputElement = (userInput) => {
    return (
        `
        <div class="input-element d-block">
            <div class="w-100 text-center text-wrap text-break">${userInput}</div>
            <button class="delete-button w-100">Delete</button>
        </div>
        `
    )
}

const addRejection = () => {
    var userInput = $("#rejection-input").val();
    if (!userInput.length > 0) {
        return;
    }
    else if (rejections.includes(userInput)) {
        alert("Rejection Texts Cannot Contain Duplicates");
        return;
    }

    rejections.push(userInput);
    $('#rejections').append(inputElement(userInput));
    $(this).val('');
}

const deleteRejection = (element) => {
    rejections = rejections.filter(function (item) {
        return item != $(element).parent('.input-element').find("div").text().replace("Delete", "")
    })
    $(element).parent(".input-element").remove();
}



const addMatch = () => {
    var userInput = $("#matches-input").find("div").find("input").val();
    console.log(userInput)
    if (!userInput.length > 0) {
        return;
    }
    else if (matches.includes(userInput)) {
        alert("Matching Texts Cannot Contain Duplicates");
        return;
    }
    matches.push(userInput);
    $('#matches').append(inputElement(userInput));
    $(this).val('');
}

const deleteMatch = (element) => {
    matches = matches.filter(function (item) {
        return item != $(element).parent('.input-element').find("div").text().replace("Delete", "")
    });
    $(element).parent('.input-element').remove();
}

$(document).ready(function () {
    $("#matches-input").on("keyup", function (event) {
        if (event.keyCode === 13) {
            event.preventDefault();
            addMatch();
        }
    });

    $("#rejections-input").on("keyup", function (event) {
        if (event.keyCode === 13) {
            event.preventDefault();
            addRejection();
        }
    });

    $("#matches").on('click', '.delete-button', function () {
        deleteMatch(this);
    });

    $("#rejections").on('click', '.delete-button', function () {
        deleteRejection(this);
    });

    $("#add-match").click(function (event) {
        event.preventDefault();
        addMatch();
    });

    $("#add-rejection").click(function (event) {
        event.preventDefault();
        addRejection();
    })
});
