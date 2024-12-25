<div>
  <div class="reading-title css-1hxq2bi">
    <h1 class="cds-1242 cds-Typography-base css-1diqjn6 cds-1244" tabindex="-1">
      Summary and Highlights
    </h1>
    <div class="css-h1vxdp"></div>
  </div>
  <div class="rc-CML" dir="auto">
    <div>
      <div
        data-track="true"
        data-track-app="open_course_home"
        data-track-page="reading_item"
        data-track-action="click"
        data-track-component="cml"
        role="presentation"
      >
        <div
          data-track="true"
          data-track-app="open_course_home"
          data-track-page="reading_item"
          data-track-action="click"
          data-track-component="cml_link"
        >
          <div data-testid="cml-viewer" class="css-1474zrz">
            <p data-text-variant="body1">
              <span
                ><span
                  >Congratulations! You have completed this lesson. At this
                  point in the course, you know that:
                </span></span
              >
            </p>
            <ul>
              <li>
                <p data-text-variant="body1">
                  <span
                    ><span
                      >A bi-gram model is a conditional probability model with
                      context size one, that is, you consider only the immediate
                      previous word to predict the next one.</span
                    ></span
                  >
                </p>
              </li>
              <li>
                <p data-text-variant="body1">
                  <span
                    ><span
                      >A trigram model is also a conditional probability
                      function and can improve on the bigram model’s limitations
                      by increasing the context size to two.</span
                    ></span
                  >
                </p>
              </li>
              <li>
                <p data-text-variant="body1">
                  <span
                    ><span
                      >The concept of a trigram can be generalized to an N-gram
                      model, which allows for an arbitrary context size.</span
                    ></span
                  >
                </p>
              </li>
              <li>
                <p data-text-variant="body1">
                  <span
                    ><span
                      >In the realm of neural networks, the context vector is
                      generally defined as the product of your context size and
                      the size of your vocabulary. Typically, this vector is not
                      computed directly but is constructed by concatenating the
                      embedding vectors.</span
                    ></span
                  >
                </p>
              </li>
              <li>
                <p data-text-variant="body1">
                  <span
                    ><span
                      >An N-gram model allows for an arbitrary context
                      size.</span
                    ></span
                  >
                </p>
              </li>
              <li>
                <p data-text-variant="body1">
                  <span
                    ><span
                      >In Pytorch, the n-gram language model is essentially a
                      classification model, using the context vector and an
                      extra hidden layer to enhance performance.</span
                    ></span
                  >
                </p>
              </li>
              <li>
                <p data-text-variant="body1">
                  <span
                    ><span
                      >The n-gram model predicts words surrounding a target by
                      incrementally shifting as a sliding window.</span
                    ></span
                  >
                </p>
              </li>
              <li>
                <p>
                  <span
                    ><span
                      >In training the model, prioritize the loss over accuracy
                      as your key performance indicator or KPI.</span
                    ></span
                  >
                </p>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div data-testid="reading-complete-container" class="css-6mc5wv">
    <div>
      <span class="rc-TooltipWrapper css-0"
        ><button
          class="cds-1207 cds-button-disableElevation cds-button-primary css-3qcgk2"
          tabindex="0"
          type="submit"
          data-testid="mark-complete"
        >
          <span class="cds-button-label">Mark as completed</span>
        </button></span
      ><button
        class="cds-1207 cds-button-disableElevation cds-button-primary css-1s4ge7s"
        tabindex="0"
        type="submit"
        data-testid="next-item"
      >
        <span class="cds-button-label">Go to next item</span>
      </button>
    </div>
    <div
      data-testid="completed-text"
      aria-live="polite"
      aria-busy="false"
      class="css-v28xtk"
    >
      <svg
        aria-hidden="true"
        fill="none"
        focusable="false"
        height="24"
        viewBox="0 0 24 24"
        width="24"
        id="cds-react-aria-998"
        class="css-1uz4a97"
      >
        <path
          fill-rule="evenodd"
          clip-rule="evenodd"
          d="M23.775 3.633L9.196 21.475.305 12.868l1.39-1.437 7.33 7.094 13.2-16.158 1.55 1.266z"
          fill="currentColor"
        ></path>
      </svg>
      <h3 class="css-1vttc58" aria-label="Reading completed">Completed</h3>
    </div>
  </div>
</div>
