let matches = [];
let rejections = [];

const inputElement = (userInput) => {
    return `
    <div class="input-element">
      <span>${userInput}</span>
      <button class="delete-button">Delete</button>
    </div>
  `;
};

const addToList = (list, input, inputId) => {
    if (input.trim().length === 0) {
        return;
    }
    if (list.includes(input)) {
        alert(`Matching Texts Cannot Contain Duplicates`);
        return;
    }
    list.push(input);
    $(`#${inputId}`).append(inputElement(input));
    $(`#${inputId} input`).val("");
};


const deleteFromList = (list, element) => {
    const text = $(element).siblings("span").text();
    list = list.filter(item => item !== text);
    $(element).parent().remove();
};

$(function () {
    $("#match-input").on("keyup", event => {
        if (event.key === "Enter") {
            event.preventDefault();
            addToList(matches, $(event.target).val(), "matches");
        }
    });

    $("#rejection-input").on("keyup", event => {
        if (event.key === "Enter") {
            event.preventDefault();
            addToList(rejections, $(event.target).val(), "rejections");
        }
    });

    $("#matches, #rejections").on("click", ".delete-button", event => {
        const parent = $(event.target).closest(".input-element").parent();
        if (parent.is("#matches")) {
            deleteFromList(matches, event.target);
        } else {
            deleteFromList(rejections, event.target);
        }
    });

    $("#add-match, #add-rejection").click(event => {
        event.preventDefault();
        const inputId = $(event.target).data("input-id");
        const inputVal = $(`#${inputId} input`).val();
        addToList(inputId === "matches" ? matches : rejections, inputVal, inputId);
    });
});